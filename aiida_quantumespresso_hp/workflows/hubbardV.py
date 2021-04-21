# -*- coding: utf-8 -*-
"""Turn-key solution to automatically compute the self-consistent 
Hubbard U+V parameters for a given structure."""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.orm import SinglefileData

from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry
from aiida_quantumespresso_hp.calculations.functions.structure_relabel_kinds import structure_relabel_kinds
from aiida_quantumespresso_hp.calculations.functions.structure_reorder_kinds import structure_reorder_kinds
from aiida_quantumespresso_hp.utils.validation import validate_structure_kind_order

PwCalculation = CalculationFactory('quantumespresso.pw')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
HpWorkChain = WorkflowFactory('quantumespresso.hp.main')


def validate_inputs(inputs, _):
    """Validate the entire inputs namespace."""
    structure_kinds = inputs['structure'].get_kind_names()


class SelfConsistentHubbardVWorkChain(WorkChain):
    """
    Workchain that will compute the self-consistent Hubbard U+V 
    parameters for a given input structure by iteratively relaxing the
    structure with the PwRelaxWorkChain and computing the Hubbard U+V
    parameters through the HpWorkChain. At this moment, a convergence 
    check of the Hubbard U+V parameters has not yet been implemented. 
    Instead, the loop is simply repeated for 'max_iterations' times and 
    then stops.

    IMPORTANT: Since there are a few bugs or missing implementations in 
    some aiida-qe routines, an initial set of "hubbard_v" parameters 
    must be provided as part of the input parameters at least for
    the SCF step in case you do not provide a parameter file right away.
    For example, an input dict for the scf part could be:

'SYSTEM': {
 'ecutrho': 1080.0,
 'ecutwfc': 90.0,
 'hubbard_v': [
     [1,1,1,1e-07],
     [2,2,1,1e-07],
     [3,3,1,1e-07],
     [9,9,1,1e-07]
 ],
 'lda_plus_u': True,
 'lda_plus_u_kind': 2,
 'u_projection_type': 'ortho-atomic',
'CONTROL': {'verbosity': 'high', 'calculation': 'scf'},
'ELECTRONS': {'conv_thr': 1e-14,
 'electron_maxstep': 150,
 'scf_must_converge': True}
})

    The procedure in each step of the convergence cycle is slightly
    different depending on the electronic and magnetic properties of 
    the system. Each cycle will roughly consist of three steps:

        * Relaxing the structure at the current Hubbard values
        * One or more SCF calculations depending on the system's electronic and magnetic properties
        * A self-consistent calculation of the Hubbard U+V parameters, restarted from the previous SCF run

    The possible options for the set of SCF calculations that have to be
    run in the second step look are:

        * Metals:
            - SCF with smearing

        * Non-magnetic insulators
            - SCF with fixed occupations

        * Magnetic insulators
            - SCF with smearing
            - SCF with fixed occupations, where total magnetization and 
              number of bands are fixed to the values found from the 
              previous SCF calculation

    When convergence is achieved, a SinglefileData node will be 
    returned containing the final Hubbard parameters.
    """

    defaults = AttributeDict({
        'qe': qe_defaults,
        'smearing_method': 'gauss',
        'smearing_degauss': 0.0036,
        'conv_thr_preconverge': 1E-6,
        'conv_thr_strictfinal': 1E-14,
        'u_projection_type_relax': 'ortho-atomic',
        'u_projection_type_scf': 'ortho-atomic',
    })

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('hubbard_file', valid_type=orm.SinglefileData,
            required=False, help='A SinglefileData node with the Hubbard parameters from a previous hp.x calculation.')
        spec.input('max_iterations', valid_type=orm.Int, default=lambda: orm.Int(3))
        spec.input('is_metal', valid_type=orm.Bool, required=False,
            help='Set this if you know whether your material is a metal or an insulator (in DFT!) to skip the reconnaissance run.')
        spec.input('do_relax', valid_type=orm.Bool, 
            default=lambda: orm.Bool(True),help='Set to false if you dont want to relax the structures between the hp.x iterations.')
        spec.input('skip_relax_till_iter', valid_type=orm.Int,
            default=lambda: orm.Int(0),help='Number of hp.x cycles that have to be completed before the ionic positions are allowed to relax.')
        spec.inputs.validator = validate_inputs
        spec.expose_inputs(PwBaseWorkChain, namespace='recon', exclude=('pw.structure',),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` that, when defined, are used for a '
                'reconnaissance SCF to determine the electronic properties of the material.'})
        spec.expose_inputs(PwRelaxWorkChain, namespace='relax', exclude=('structure',),
            namespace_options={'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain` that, when defined, will iteratively relax the structure.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='scf', exclude=('pw.structure',))
        spec.expose_inputs(HpWorkChain, namespace='hubbard', exclude=('hp.parent_scf',))
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_recon)(
                cls.run_recon,
                cls.inspect_recon,
            ),
            while_(cls.should_run_iteration)(
                cls.update_iteration,
                if_(cls.should_run_relax)(
                    cls.run_relax,
                    cls.inspect_relax,
                ),
                if_(cls.is_metal)(  
                    cls.run_scf_smearing,
                ).elif_(cls.is_magnetic)(
                    cls.run_scf_smearing,
                    cls.run_scf_fixed_magnetic,
                ).else_(
                    cls.run_scf_fixed,
                ),
                cls.inspect_scf,
                cls.run_hp,
                cls.inspect_hp,
            ),
            cls.run_results,
        )
        spec.output('structure', valid_type=orm.StructureData, required=False,
            help='The final relaxed structure, only if relax inputs were defined.')
        spec.output('hubbard_parameters', valid_type=orm.SinglefileData,
            help='The final Hubbard parameters.')
        spec.exit_code(330, 'ERROR_FAILED_TO_DETERMINE_PSEUDO_POTENTIAL',
            message='Failed to determine the correct pseudo potential after the structure changed its kind names.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RECON',
            message='The reconnaissance PwBaseWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='The PwRelaxWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The scf PwBaseWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_HP',
            message='The HpWorkChain sub process failed in iteration {iteration}')

    def setup(self):
        """Set up the context."""
        try:
            self.ctx.current_hubbard_file = self.inputs.hubbard_file
            self.ctx.has_hubbard_file = True
        except AttributeError:
            self.report('No initial Hubbard file was provided. Continuing...')
            self.ctx.has_hubbard_file = False

        self.ctx.max_iterations = self.inputs.max_iterations.value
        self.ctx.current_structure = self.inputs.structure
        self.ctx.is_converged = False
        self.ctx.is_magnetic = None
        self.ctx.iteration = 0

    def validate_inputs(self):
        """Validate inputs."""
        # Determine whether the system is to be treated as magnetic
        parameters = self.inputs.scf.pw.parameters.get_dict()
        if parameters.get('SYSTEM', {}).get('nspin', self.defaults.qe.nspin) != 1:
            self.report('system is treated to be magnetic because `nspin != 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = True
        else:
            self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = False

    def should_run_recon(self):
        """Return whether a recon calculation needs to be run, which is true if 'is_metal' has not been specified in inputs.
           This serves to find out the wheter the metal is conducting or insulating."""
        try:
            self.ctx.is_metal = bool(self.inputs.is_metal)
            if self.ctx.is_metal:
                self.report('The system will be treated as a metal due to user input.')
            if not self.ctx.is_metal:
                self.report('The system will be treated as an insulator due to user input.')
            return False
        except AttributeError:
            self.report('It is unknown whether the material is metallic or insulatating. Will carry out a reconnaisance run...')
            return True

    def should_run_relax(self):
        """Return whether a relax calculation needs to be run.
           If 'skip_relax_till_iter' is set to N, do not carry out the relaxation until the Nth hp.x cycle 
           finished (to avoid stabilization of unphysical structures due to an'overshooting' of U and V).
           Relaxations will also not be carried out if 'do_relax' has been set to False or if no relax inputs
           have been provided.
        """
        if int(self.inputs.skip_relax_till_iter) >= (self.ctx.iteration):
            return False
        else:
            return (('relax' in self.inputs) and bool(self.inputs.do_relax))

    def should_run_iteration(self):
        """Return whether a new process should be run.
           This is the case as long as the Hubbard parameters have not yet converged 
           and the maximum number of restarts has not yet been exceeded.
        """
        return self.ctx.iteration < self.ctx.max_iterations

    def update_iteration(self):
        """Update the current iteration index counter."""
        self.ctx.iteration += 1

    def has_hubbard_file(self):
        """Return whether a parameter file has been provided"""
        return self.ctx.has_hubbard_file

    def is_metal(self):
        """Return whether the current structure is a metal."""
        return self.ctx.is_metal

    def is_magnetic(self):
        """Return whether the current structure is magnetic."""
        return self.ctx.is_magnetic

    def get_inputs(self, cls, namespace):
        """Return the inputs for one of the subprocesses whose inputs are exposed in the given namespace.

        :param cls: the process class of the subprocess
        :param namespace: namespace into which the inputs are exposed.
        :return: dictionary with inputs.
        """
        inputs = AttributeDict(self.exposed_inputs(cls, namespace=namespace))

        try:
            pseudos = self.get_pseudos()
        except ValueError:
            return self.exit_codes.ERROR_FAILED_TO_DETERMINE_PSEUDO_POTENTIAL

        if cls is PwBaseWorkChain and namespace in ['recon', 'scf']:
            inputs.pw.pseudos = pseudos
            inputs.pw.structure = self.ctx.current_structure
            inputs.pw.parameters = inputs.pw.parameters.get_dict()
            inputs.pw.parameters.setdefault('CONTROL', {})
            inputs.pw.parameters.setdefault('SYSTEM', {})
            inputs.pw.parameters.setdefault('ELECTRONS', {})
        elif cls is PwRelaxWorkChain and namespace == 'relax':
            inputs.structure = self.ctx.current_structure
            inputs.base.pw.pseudos = pseudos
            inputs.base.pw.parameters = inputs.base.pw.parameters.get_dict()
            inputs.base.pw.parameters.setdefault('CONTROL', {})
            inputs.base.pw.parameters.setdefault('SYSTEM', {})
            inputs.base.pw.parameters.setdefault('ELECTRONS', {})

        return inputs

    def get_pseudos(self):
        """Return the mapping of pseudos based on the current structure.

        .. note:: this is necessary because during the workchain the kind names of the structure can change, meaning the
            mapping of the pseudos that is to be passed to the subprocesses also may have to change, since the keys are
            based on the kind names of the structure.

        :return: dictionary of pseudos where the keys are the kindnames of ``self.ctx.current_structure``.
        """
        import re

        results = {}
        pseudos = self.inputs.scf.pw.pseudos

        for kind in self.ctx.current_structure.kinds:
            for key, pseudo in pseudos.items():
                symbol = re.sub(r'\d', '', key)
                if re.match(r'{}[0-9]*'.format(kind.symbol), symbol):
                    results[kind.name] = pseudo
                    break
            else:
                raise ValueError(f'could not find the pseudo from inputs.scf.pw.pseudos for kind `{kind}`.')

        return results

    def run_recon(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation.

        This runs a simple scf cycle with a few steps with smearing turned on to determine whether the system is most
        likely a metal or an insulator. This step is required because the metallicity of the systems determines how the
        relaxation calculations in the convergence cycle have to be performed.
        """
        inputs = self.get_inputs(PwBaseWorkChain, 'recon')
        inputs.pw.parameters.setdefault('CONTROL', {})['calculation'] = 'scf'
        inputs.pw.parameters.setdefault('ELECTRONS', {})['scf_must_converge'] = False
        inputs.pw.parameters.setdefault('ELECTRONS', {})['electron_maxstep'] = 40
        inputs.pw.parameters.setdefault('SYSTEM', {})['occupations'] = 'smearing'
        inputs.pw.parameters.setdefault('SYSTEM', {})['smearing'] = self.defaults.smearing_method
        inputs.pw.parameters.setdefault('SYSTEM', {})['degauss'] = self.defaults.smearing_degauss
        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)
        inputs.metadata.call_link_label = 'recon'

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching reconnaissance PwBaseWorkChain<{running.pk}>')
        return ToContext(workchain_recon=running)

    def inspect_recon(self):
        """Verify that the reconnaissance PwBaseWorkChain finished successfully and determine whether the
         material is a metal or an insulator.
         """
        workchain = self.ctx.workchain_recon

        if not workchain.is_finished_ok:
            self.report(f'reconnaissance PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RECON.format(iteration=self.ctx.iteration)

        bands = workchain.outputs.output_band
        parameters = workchain.outputs.output_parameters.get_dict()
        number_electrons = parameters['number_of_electrons']

        is_insulator, _ = find_bandgap(bands, number_electrons=number_electrons)

        if is_insulator:
            self.report('system is determined to be an insulator')
            self.ctx.is_metal = False
        else:
            self.report('system is determined to be a metal')
            self.ctx.is_metal = True

    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation."""
        inputs = self.get_inputs(PwRelaxWorkChain, 'relax')
        parameters = inputs.base.pw.parameters

        parameters['SYSTEM']['u_projection_type'] = self.defaults.u_projection_type_relax
        parameters['SYSTEM']['lda_plus_u'] = True
        parameters['SYSTEM']['lda_plus_u_kind'] = 2

        if not self.ctx.has_hubbard_file and not 'hubbard_v' in parameters['SYSTEM']:
           parameters['SYSTEM']['lda_plus_u'] = False
        elif self.ctx.has_hubbard_file:
            parameters['SYSTEM'].pop('hubbard_v', None)
            parameters['SYSTEM']['Hubbard_parameters'] = 'file'
            inputs.base.pw.hubbard_file = self.ctx.current_hubbard_file
            
        inputs.base.pw.parameters = orm.Dict(dict=parameters)
        inputs.metadata.call_link_label = 'iteration_{:02d}_relax'.format(self.ctx.iteration)

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> iteration #{self.ctx.iteration}')
        return ToContext(workchains_relax=append_(running))

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully and check if the system is now metallic or insulating."""
        workchain = self.ctx.workchains_relax[-1]

        if not workchain.is_finished_ok:
            self.report(f'PwRelaxWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX.format(iteration=self.ctx.iteration)

        self.ctx.current_structure = workchain.outputs.output_structure
        
        bands = workchain.outputs.output_band
        parameters = workchain.outputs.output_parameters.get_dict()
        number_electrons = parameters['number_of_electrons']

        is_insulator, _ = find_bandgap(bands, number_electrons=number_electrons)

        if is_insulator and self.ctx.is_metal:
            self.report('After relaxation, system is now determined to be an insulator')
            self.ctx.is_metal = False
        elif not is_insulator and not self.ctx.is_metal:
            self.report('After relaxation, system is now determined to be a metal')
            self.ctx.is_metal = True


    def run_scf_fixed(self):
        """Run an scf `PwBaseWorkChain` with fixed occupations."""
        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'
        inputs.pw.parameters['SYSTEM']['occupations'] = 'fixed'
        inputs.pw.parameters['SYSTEM'].pop('degauss', None)
        inputs.pw.parameters['SYSTEM'].pop('smearing', None) 
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = inputs.pw.parameters['SYSTEM'].get('u_projection_type', self.defaults.u_projection_type_scf)
        inputs.pw.parameters['SYSTEM']['lda_plus_u'] = True
        inputs.pw.parameters['SYSTEM']['lda_plus_u_kind'] = 2
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = inputs.pw.parameters['ELECTRONS'].get('conv_thr', self.defaults.conv_thr_strictfinal)
            #If a Hubbard file was provided, use it. Otherwise, use the hubbard_v dict as input.
        if self.ctx.has_hubbard_file:
            inputs.pw.parameters['SYSTEM'].pop('hubbard_v', None)
            inputs.pw.parameters['SYSTEM']['Hubbard_parameters'] = 'file'
            inputs.pw.hubbard_file = self.ctx.current_hubbard_file

        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)
        inputs.metadata.call_link_label = 'iteration_{:02d}_scf_fixed'.format(self.ctx.iteration)

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> with fixed occupations')
        return ToContext(workchains_scf=append_(running))

    def run_scf_smearing(self):
        """Run an scf `PwBaseWorkChain` with smeared occupations"""
        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'
        inputs.pw.parameters['SYSTEM']['occupations'] = 'smearing'
        inputs.pw.parameters['SYSTEM']['smearing'] = inputs.pw.parameters['SYSTEM'].get('smearing', self.defaults.smearing_method)
        inputs.pw.parameters['SYSTEM']['degauss'] = inputs.pw.parameters['SYSTEM'].get('degauss', self.defaults.smearing_degauss)
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = inputs.pw.parameters['SYSTEM'].get('u_projection_type', self.defaults.u_projection_type_scf)
        inputs.pw.parameters['SYSTEM']['lda_plus_u'] = True
        inputs.pw.parameters['SYSTEM']['lda_plus_u_kind'] = 2
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = inputs.pw.parameters['ELECTRONS'].get('conv_thr', self.defaults.conv_thr_strictfinal)

        #If the system is metallic, you cannot fix the magnetization, otherwise hp.x will fail
        if self.ctx.is_metal:
            inputs.pw.parameters['SYSTEM'].pop('tot_magnetization', None)
        #Use a very small smearing for insulators and semiconductors to avoid 'fake occupations'
        else:
            inputs.pw.parameters['SYSTEM']['degauss'] = 0.00072

        #If a Hubbard file was provided, use it. Otherwise, use the hubbard_v dict as input.
        if self.ctx.has_hubbard_file:
            inputs.pw.parameters['SYSTEM'].pop('hubbard_v', None)
            inputs.pw.parameters['SYSTEM']['Hubbard_parameters'] = 'file'
            inputs.pw.hubbard_file = self.ctx.current_hubbard_file

        inputs.metadata.call_link_label = 'iteration_{:02d}_scf_smearing'.format(self.ctx.iteration)
        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launching PwBaseWorkChain<{running.pk}> with smeared occupations')
        return ToContext(workchains_scf=append_(running))

    def run_scf_fixed_magnetic(self):
        """Run an scf `PwBaseWorkChain` with fixed occupations restarting from the previous calculation.

        The number of bands and total magnetization are set according to those of the previous calculation that was
        run with smeared occupations.
        """
        previous_workchain = self.ctx.workchains_scf[-1]
        previous_parameters = previous_workchain.outputs.output_parameters
		
        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        inputs.pw.parent_folder = previous_workchain.outputs.remote_folder
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'
        inputs.pw.parameters['CONTROL']['restart_mode'] = 'from_scratch'
        inputs.pw.parameters['SYSTEM']['occupations'] = 'fixed'
        inputs.pw.parameters['SYSTEM'].pop('degauss', None)
        inputs.pw.parameters['SYSTEM'].pop('smearing', None)
        inputs.pw.parameters['SYSTEM'].pop('starting_magnetization', None)
        inputs.pw.parameters['SYSTEM']['nbnd'] = previous_parameters.get_dict()['number_of_bands']
        inputs.pw.parameters['SYSTEM']['tot_magnetization'] = abs(round(previous_parameters.get_dict()['total_magnetization']))
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = inputs.pw.parameters['SYSTEM'].get('u_projection_type', self.defaults.u_projection_type_scf)
        inputs.pw.parameters['SYSTEM']['lda_plus_u'] = True
        inputs.pw.parameters['SYSTEM']['lda_plus_u_kind'] = 2
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = inputs.pw.parameters['ELECTRONS'].get('conv_thr', self.defaults.conv_thr_strictfinal)
        inputs.pw.parameters['ELECTRONS']['startingpot'] = 'file'
        
        #If a Hubbard file was provided, use it. Otherwise, use the hubbard_v dict as input.
        if self.ctx.has_hubbard_file:
            inputs.pw.parameters['SYSTEM'].pop('hubbard_v', None)
            inputs.pw.parameters['SYSTEM']['Hubbard_parameters'] = 'file'
            inputs.pw.hubbard_file = self.ctx.current_hubbard_file

        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)
        inputs.metadata.call_link_label = 'iteration_{:02d}_scf_fixed_magnetic'.format(self.ctx.iteration)

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> with fixed occupations, bands and total magnetization')
        return ToContext(workchains_scf=append_(running))

    def inspect_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchains_scf[-1]

        if not workchain.is_finished_ok:
            self.report(f'scf in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF.format(iteration=self.ctx.iteration)

    def run_hp(self):
        """
        Run the HpWorkChain restarting from the last completed scf calculation
        """
        workchain = self.ctx.workchains_scf[-1]

        inputs = self.get_inputs(HpWorkChain, 'hubbard')
        inputs.hp.parent_scf = workchain.outputs.remote_folder
        inputs.metadata.call_link_label = 'iteration_{:02d}_hp'.format(self.ctx.iteration)

        running = self.submit(HpWorkChain, **inputs)

        self.report(f'launching HpWorkChain<{running.pk}> iteration #{self.ctx.iteration}')
        return ToContext(workchains_hp=append_(running))

    def inspect_hp(self):
        """
        Analyze the last completed HpWorkChain. We check the current Hubbard U parameters and compare those with
        the values computed in the previous iteration. If the difference for all Hubbard sites is smaller than
        the tolerance, the calculation is considered to be converged.
        """
        workchain = self.ctx.workchains_hp[-1]

        if not workchain.is_finished_ok:
            self.report(f'hp.x in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_HP.format(iteration=self.ctx.iteration)
        else:
            self.ctx.has_hubbard_file = True
            self.ctx.current_hubbard_file = workchain.outputs.hubbard_parameters

    def run_results(self):
        """Attach the final converged Hubbard parameters and the corresponding structure."""
        self.report(f'Hubbard parameters have been computed for {self.ctx.iteration} iterations')
        self.out('structure', self.ctx.current_structure)
        self.out('hubbard_parameters', self.ctx.workchains_hp[-1].outputs.hubbard_parameters)
