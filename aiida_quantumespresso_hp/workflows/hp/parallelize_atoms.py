# -*- coding: utf-8 -*-
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, CalculationFactory, WorkflowFactory
from aiida.orm.data.base import Bool, Int
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.work.workchain import WorkChain, ToContext, append_
from aiida.work.workfunctions import workfunction


PwCalculation = CalculationFactory('quantumespresso.pw')
HpCalculation = CalculationFactory('quantumespresso.hp')
HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')


class HpParallelizeAtomsWorkChain(WorkChain):
    """
    Workchain to launch a Quantum Espresso hp.x calculation for a completed PwCalculation
    while parallelizing the calculation over the Hubbard atoms
    """

    ERROR_CHILD_WORKCHAIN_FAILED = 100

    @classmethod
    def define(cls, spec):
        super(HpParallelizeAtomsWorkChain, cls).define(spec)
        spec.expose_inputs(HpBaseWorkChain)
        spec.outline(
            cls.run_init,
            cls.run_atoms,
            cls.run_collect,
            cls.run_final,
            cls.results
        )
        spec.expose_outputs(HpBaseWorkChain)

    def run_init(self):
        """
        Run an initialization HpCalculatio that will only perform the symmetry analysis
        and determine which kinds are to be perturbed. This information is parsed and can
        be used to determine exactly how many HpBaseWorkChains have to be launched
        """
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.only_initialization = Bool(True)

        running = self.submit(HpBaseWorkChain, **inputs)

        self.report('launching initialization HpBaseWorkChain<{}>'.format(running.pk))

        return ToContext(initialization=running)

    def run_atoms(self):
        """
        Run a separate HpBaseWorkChain for each of the defined Hubbard atoms
        """
        workchain = self.ctx.initialization

        if not workchain.is_finished_ok:
            self.report('initialization workchain<{}> failed with finish status {}, aborting...'
                .format(workchain.pk, workchain.finish_status))
            return self.ERROR_CHILD_WORKCHAIN_FAILED

        output_params = workchain.out.output_parameters.get_dict()
        hubbard_sites = output_params['hubbard_sites']

        for site_index, site_kind in hubbard_sites.iteritems():

            do_only_key = 'do_one_only({})'.format(site_index)

            inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
            inputs.parameters = inputs.parameters.get_dict()
            inputs.parameters['INPUTHP'][do_only_key] = True
            inputs.parameters = ParameterData(dict=inputs.parameters)

            running = self.submit(HpBaseWorkChain, **inputs)

            self.report('launching HpBaseWorkChain<{}> for atomic site {} of kind {}'.format(running.pk, site_index, site_kind))
            self.to_context(workchains=append_(running))

    def run_collect(self):
        """
        Collect all the retrieved folders of the launched HpBaseWorkChain and merge them into
        a single FolderData object that will be used for the final HpCalculation
        """
        retrieved_folders = {}

        for workchain in self.ctx.workchains:

            if not workchain.is_finished_ok:
                self.report('child workchain<{}> failed with finish status {}, aborting...'
                    .format(workchain.pk, workchain.finish_status))
                return self.ERROR_CHILD_WORKCHAIN_FAILED

            retrieved = workchain.out.retrieved
            output_params = workchain.out.output_parameters
            atomic_site_index = output_params.get_dict()['hubbard_sites'].keys()[0]
            retrieved_folders[atomic_site_index] = retrieved

        self.ctx.merged_retrieved = recollect_atomic_calculations(**retrieved_folders)

    def run_final(self):
        """
        Perform the final HpCalculation to collect the various components of the chi matrices
        """
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.parameters = inputs.parameters.get_dict()
        inputs.parameters['INPUTHP']['collect_chi'] = True
        inputs.parameters = ParameterData(dict=inputs.parameters)
        inputs.parent_folder = self.ctx.merged_retrieved
        inputs.pop('parent_calculation', None)

        running = self.submit(HpBaseWorkChain, **inputs)

        self.report('launching HpBaseWorkChain<{}> to collect matrices'.format(running.pk))
        self.to_context(workchains=append_(running))

    def results(self):
        """
        Retrieve the results from the final matrix collection workchain
        """
        self.out_many(self.exposed_outputs(self.ctx.workchains[-1], HpBaseWorkChain))


@workfunction
def recollect_atomic_calculations(**kwargs):
    """
    Collect dynamical matrix files into a single folder, putting a different number at the end of
    each final dynamical matrix file, obtained from the input link, which corresponds to its place
    in the list of q-points originally generated by distribute_qpoints.
    
    :param kwargs: keys are the string representation of the hubbard atom index and the value is the 
        corresponding retrieved folder object.
    :return: FolderData object containing the perturbation files of the computed HpBaseWorkChain
    """
    import os
    import errno

    output_folder_sub = HpCalculation._OUTPUT_SUBFOLDER
    output_folder_raw = HpCalculation._FOLDER_RAW
    output_prefix = HpCalculation()._PREFIX

    # Initialize the merged folder, by creating the subdirectory for the perturbation files
    merged_folder = FolderData()
    folder_path = os.path.normpath(merged_folder.get_abs_path('.'))
    output_path = os.path.join(folder_path, output_folder_raw)

    try:
        os.makedirs(output_path)
    except OSError as error:
        if error.errno == errno.EEXIST and os.path.isdir(output_path):
            pass
        else:
            raise

    for atomic_site_index, retrieved_folder in kwargs.iteritems():
        filepath = os.path.join(output_folder_raw, '{}.chi.pert_{}.dat'.format(output_prefix, atomic_site_index))
        filepath_src = retrieved_folder.get_abs_path(filepath)
        filepath_dst = filepath
        merged_folder.add_path(filepath_src, filepath_dst)

    # TODO: currently the Hp code requires the .save folder that is written by the original
    # PwCalculation, for the final post-processing matrix collection step. It doesn't really need all
    # the information contained in that folder, and requiring it means, copying it from remote to a
    # local folder and then reuploading it to remote folder. This is unnecessarily heavy
    retrieved_folder = kwargs.values()[0]
    dirpath = os.path.join(output_folder_sub, output_prefix + '.save')
    dirpath_src = retrieved_folder.get_abs_path(dirpath)
    dirpath_dst = dirpath
    merged_folder.add_path(dirpath_src, dirpath_dst)

    retrieved_folder = kwargs.values()[0]
    filepath = os.path.join(output_folder_sub, output_prefix + '.occup')
    filepath_src = retrieved_folder.get_abs_path(filepath)
    filepath_dst = filepath
    merged_folder.add_path(filepath_src, filepath_dst)
    
    return merged_folder