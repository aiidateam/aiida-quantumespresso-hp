# -*- coding: utf-8 -*-
import copy
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.base import Bool, Int, Str
from aiida.orm.data.upf import UpfData, get_pseudos_from_structure
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.work.run import submit
from aiida.work.workchain import WorkChain, ToContext, append_
from aiida.work.workfunction import workfunction
from aiida_quantumespresso_uscf.workflows.uscf.base import UscfBaseWorkChain

PwCalculation = CalculationFactory('quantumespresso.pw')
UscfCalculation = CalculationFactory('quantumespresso.uscf')

class UscfParallelizeAtomsWorkChain(WorkChain):
    """
    """
    def __init__(self, *args, **kwargs):
        super(UscfParallelizeAtomsWorkChain, self).__init__(*args, **kwargs)

    @classmethod
    def define(cls, spec):
        super(UscfParallelizeAtomsWorkChain, cls).define(spec)
        spec.input('code', valid_type=Code)
        spec.input('parent_calculation', valid_type=PwCalculation)
        spec.input('qpoints', valid_type=KpointsData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('settings', valid_type=ParameterData)
        spec.input('options', valid_type=ParameterData)
        spec.input('max_iterations', valid_type=Int, default=Int(10))
        spec.outline(
            cls.setup,
            cls.run_init,
            cls.run_atoms,
            cls.run_collect,
            cls.run_final,
            cls.run_results
        )
        spec.dynamic_output()

    def setup(self):
        """
        Determine the set of Hubbard U atoms and the raw inputs dictionary
        """
        parameters = self.inputs.parent_calculation.inp.parameters
        self.ctx.hubbard_atoms = parameters.get_dict()['SYSTEM']['hubbard_u']

        self.ctx.raw_inputs = {
            'code': self.inputs.code,
            'qpoints': self.inputs.qpoints,
            'parameters': self.inputs.parameters.get_dict(),
            'parent_folder': self.inputs.parent_calculation.out.remote_folder,
            'settings': self.inputs.settings,
            'options': self.inputs.options,
            'max_iterations': self.inputs.max_iterations,
        }

    def run_init(self):
        """
        Run an initialization UscfCalculatio that will only perform the symmetry analysis
        and determine which kinds are to be perturbed. This information is parsed and can
        be used to determine exactly how many UscfBaseWorkChains have to be launched
        """
        inputs = {
            'code': self.inputs.code,
            'qpoints': self.inputs.qpoints,
            'parameters': self.inputs.parameters.get_dict(),
            'parent_folder': self.inputs.parent_calculation.out.remote_folder,
            'settings': self.inputs.settings,
            '_options': self.inputs.options.get_dict(),
        }

        inputs['parameters']['INPUTUSCF']['determine_num_pert_only'] = True
        inputs['parameters'] = ParameterData(dict=inputs['parameters'])

        process = UscfCalculation.process()
        running = submit(process, **inputs)

        self.report('launching initialization UscfCalculation<{}>'.format(running.pid))

        return ToContext(initialization=running)

    def run_atoms(self):
        """
        Run a separate UscfBaseWorkChain for each of the defined Hubbard atoms
        """
        output_params = self.ctx.initialization.out.parameters.get_dict()
        hubbard_sites = output_params['hubbard_sites']

        for site_index, site_kind in hubbard_sites.iteritems():

            do_only_key = 'do_one_only({})'.format(site_index)

            inputs = copy.deepcopy(self.ctx.raw_inputs)
            inputs['parameters']['INPUTUSCF'][do_only_key] = True
            inputs['parameters'] = ParameterData(dict=inputs['parameters'])

            running = submit(UscfBaseWorkChain, **inputs)

            self.report('launching UscfBaseWorkChain<{}> for atomic site {} of kind {}'.format(running.pid, site_index, site_kind))
            self.to_context(workchains=append_(running))

    def run_collect(self):
        """
        Collect all the retrieved folders of the launched UscfBaseWorkChain and merge them into
        a single FolderData object that will be used for the final UscfCalculation
        """
        retrieved_folders = {}

        for workchain in self.ctx.workchains:
            retrieved = workchain.out.retrieved
            output_params = workchain.out.parameters
            atomic_site_index = output_params.get_dict()['hubbard_sites'].keys()[0]
            retrieved_folders[atomic_site_index] = retrieved

        self.ctx.merged_retrieved = recollect_atomic_calculations(**retrieved_folders)

    def run_final(self):
        """
        Perform the final UscfCalculation to collect the various components of the chi matrices
        """
        inputs = copy.deepcopy(self.ctx.raw_inputs)
        inputs['parent_folder'] = self.ctx.merged_retrieved
        inputs['parameters']['INPUTUSCF']['collect_chi'] = True
        inputs['parameters'] = ParameterData(dict=inputs['parameters'])

        running = submit(UscfBaseWorkChain, **inputs)

        self.report('launching UscfBaseWorkChain<{}> to collect matrices'.format(running.pid))
        self.to_context(workchains=append_(running))

    def run_results(self):
        """
        Retrieve the results from the final matrix collection calculation
        """
        workchains = self.ctx.workchains[-1]

        # We expect the last workchain, which was a matrix collecting calculation, to
        # have all the output links. If not something must have gone wrong
        for link in ['retrieved', 'parameters', 'chi', 'hubbard', 'matrices']:
            if not link in workchains.out:
                self.abort_nowait("final collecting workchain is missing expected output link '{}'".format(link))
            else:
                self.out(link, workchains.out[link])

        self.report('workchain completed successfully')


@workfunction
def recollect_atomic_calculations(**kwargs):
    """
    Collect dynamical matrix files into a single folder, putting a different number at the end of
    each final dynamical matrix file, obtained from the input link, which corresponds to its place
    in the list of q-points originally generated by distribute_qpoints.
    
    :param kwargs: keys are the string representation of the hubbard atom index and the value is the 
        corresponding retrieved folder object.
    :return: FolderData object containing the perturbation files of the computed UscfBaseWorkChain
    """
    import os
    import errno

    output_folder_sub = UscfCalculation._OUTPUT_SUBFOLDER
    output_folder_ph0 = UscfCalculation._FOLDER_PH0
    output_prefix = UscfCalculation()._PREFIX

    # Initialize the merged folder, by creating the subdirectory for the perturbation files
    merged_folder = FolderData()
    folder_path = os.path.normpath(merged_folder.get_abs_path('.'))
    output_path = os.path.join(folder_path, output_folder_ph0)

    try:
        os.makedirs(output_path)
    except OSError as error:
        if error.errno == errno.EEXIST and os.path.isdir(output_path):
            pass
        else:
            raise

    for atomic_site_index, retrieved_folder in kwargs.iteritems():
        filepath = os.path.join(output_folder_ph0, '{}.chi.pert_{}.dat'.format(output_prefix, atomic_site_index))
        filepath_src = retrieved_folder.get_abs_path(filepath)
        filepath_dst = filepath
        merged_folder.add_path(filepath_src, filepath_dst)

    # TODO: currently the Uscf code requires the .save folder that is written by the original
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