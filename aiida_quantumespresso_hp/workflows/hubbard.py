# -*- coding: utf-8 -*-
"""Turn-key solution to automatically compute the self-consistent Hubbard parameters for a given structure."""
import yaml
from importlib_resources import files

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
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
    hubbard_u_kinds = list(inputs['hubbard_u'].get_dict().keys())

    if not hubbard_u_kinds:
        return 'need to define a starting Hubbard U value for at least one kind.'

    if not set(hubbard_u_kinds).issubset(structure_kinds):
        return 'kinds specified in starting Hubbard U values is not a strict subset of the structure kinds.'


class SelfConsistentHubbardWorkChain(ProtocolMixin, WorkChain):
    """
    Workchain that for a given input structure will compute the self-consistent Hubbard U parameters
    by iteratively relaxing the structure with the PwRelaxWorkChain and computing the Hubbard U
    parameters through the HpWorkChain, until the Hubbard U values are converged within a certain tolerance.

    The procedure in each step of the convergence cycle is slightly different depending on the electronic and
    magnetic properties of the system. Each cycle will roughly consist of three steps:

        * Relaxing the structure at the current Hubbard U values
        * One or more SCF calculations depending on the system's electronic and magnetic properties
        * A self-consistent calculation of the Hubbard U parameters, restarted from the previous SCF run

    The possible options for the set of SCF calculations that have to be run in the second step look are:

        * Metals:
            - SCF with smearing

        * Non-magnetic insulators
            - SCF with fixed occupations

        * Magnetic insulators
            - SCF with smearing
            - SCF with fixed occupations, where total magnetization and number of bands are fixed
              to the values found from the previous SCF calculation

    When convergence is achieved a Dict node will be returned containing the final converged
    Hubbard U parameters.
    """

    # pylint: disable=too-many-public-methods

    defaults = AttributeDict({
        'qe': qe_defaults,
        'smearing_method': 'marzari-vanderbilt',
        'smearing_degauss': 0.02,
        'conv_thr_preconverge': 1E-10,
        'conv_thr_strictfinal': 1E-15,
        'u_projection_type_relax': 'atomic',
        'u_projection_type_scf': 'ortho-atomic',
    })

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('hubbard_u', valid_type=orm.Dict)
        spec.input('tolerance', valid_type=orm.Float, default=lambda: orm.Float(0.1))
        spec.input('max_iterations', valid_type=orm.Int, default=lambda: orm.Int(5))
        spec.input('meta_convergence', valid_type=orm.Bool, default=lambda: orm.Bool(False))
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
                if_(cls.is_metal)(  # pylint: disable=no-member
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
        spec.output('hubbard', valid_type=orm.Dict,
            help='The final converged Hubbard U parameters.')
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

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        hp_code,
        structure,
        protocol=None,
        overrides=None,
        **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        """
        pw_args = (pw_code, structure, protocol)
        inputs = cls.get_protocol_inputs(protocol, overrides)

        recon = PwBaseWorkChain.get_builder_from_protocol(*pw_args, overrides=inputs.get('recon', None), **kwargs)
        recon.pw.pop('structure', None)
        relax = PwRelaxWorkChain.get_builder_from_protocol(*pw_args, overrides=inputs.get('relax', None), **kwargs)
        relax.pop('structure', None)
        scf = PwBaseWorkChain.get_builder_from_protocol(*pw_args, overrides=inputs.get('scf', None), **kwargs)
        scf.pw.pop('structure', None)

        hubbard = HpWorkChain.get_builder_from_protocol(
            code=hp_code,
            protocol=protocol,
            overrides=inputs.get('scf', None), 
            **kwargs
        )

        hubbard_u = inputs.get('hubbard_u', None)
        hubbard_u = hubbard_u or cls._load_hubbard_u(structure)

        builder = cls.get_builder()
        builder.structure = structure
        builder.hubbard_u = hubbard_u
        builder.recon = recon
        builder.relax = relax
        builder.scf = scf
        builder.hubbard = hubbard

        return builder

    def _load_hubbard_u(structure):
        """Load the default values for the initial hubbard U setting."""
        import aiida_quantumespresso_hp.workflows.protocols

        with files(aiida_quantumespresso_hp.workflows.protocols).joinpath('hubbard_u.yaml').open() as file:
            hubbard_values = yaml.safe_load(file)

        hubbard_u = {}

        for kind in structure.kinds:
            hubbard_u[kind.symbol] = hubbard_values.get(kind.symbol, 0)

        return orm.Dict(dict=hubbard_u)

    def setup(self):
        """Set up the context."""
        self.ctx.max_iterations = self.inputs.max_iterations.value
        self.ctx.current_structure = self.inputs.structure
        self.ctx.current_hubbard_u = self.inputs.hubbard_u.get_dict()
        self.ctx.is_converged = False
        self.ctx.is_magnetic = None
        self.ctx.is_metal = None
        self.ctx.iteration = 0

    def validate_inputs(self):
        """Validate inputs."""
        structure = self.inputs.structure
        hubbard_u = self.inputs.hubbard_u

        try:
            validate_structure_kind_order(structure, list(hubbard_u.get_dict().keys()))
        except ValueError:
            self.report('structure has incorrect kind order, reordering...')
            self.ctx.current_structure = structure_reorder_kinds(structure, hubbard_u)
            self.report(f'reordered StructureData<{structure.pk}>')

        # Determine whether the system is to be treated as magnetic
        parameters = self.inputs.scf.pw.parameters.get_dict()
        if parameters.get('SYSTEM', {}).get('nspin', self.defaults.qe.nspin) != 1:
            self.report('system is treated to be magnetic because `nspin != 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = True
        else:
            self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = False

    def should_run_recon(self):
        """Return whether a recon calculation needs to be run, which is true if `recon` is specified in inputs."""
        return 'recon' in self.inputs

    def should_run_relax(self):
        """Return whether a relax calculation needs to be run, which is true if `relax` is specified in inputs."""
        return 'relax' in self.inputs

    def should_run_iteration(self):
        """Return whether a new process should be run.

        This is the case as long as the Hubbard parameters have not yet converged and the maximum number of restarts has
        not yet been exceeded.
        """
        return not self.ctx.is_converged and self.ctx.iteration < self.ctx.max_iterations

    def update_iteration(self):
        """Update the current iteration index counter."""
        self.ctx.iteration += 1

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
            inputs.pw.parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
        elif cls is PwRelaxWorkChain and namespace == 'relax':
            inputs.structure = self.ctx.current_structure
            inputs.base.pw.pseudos = pseudos
            inputs.base.pw.parameters = inputs.base.pw.parameters.get_dict()
            inputs.base.pw.parameters.setdefault('CONTROL', {})
            inputs.base.pw.parameters.setdefault('SYSTEM', {})
            inputs.base.pw.parameters.setdefault('ELECTRONS', {})
            inputs.base.pw.parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u

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
        pseudos = self.inputs.recon.pw.pseudos

        for kind in self.ctx.current_structure.kinds:
            for key, pseudo in pseudos.items():
                symbol = re.sub(r'\d', '', key)
                if re.match(r'{}[0-9]*'.format(kind.symbol), symbol):
                    results[kind.name] = pseudo
                    break
            else:
                raise ValueError(f'could not find the pseudo from inputs.recon.pw.pseudos for kind `{kind}`.')

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
        inputs.pw.parameters.setdefault('ELECTRONS', {})['electron_maxstep'] = 10
        inputs.pw.parameters.setdefault('SYSTEM', {})['occupations'] = 'smearing'
        inputs.pw.parameters.setdefault('SYSTEM', {})['smearing'] = self.defaults.smearing_method
        inputs.pw.parameters.setdefault('SYSTEM', {})['degauss'] = self.defaults.smearing_degauss
        inputs.pw.parameters.setdefault('SYSTEM', {}).pop('lda_plus_u', None)
        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)
        inputs.metadata.call_link_label = 'recon'

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching reconnaissance PwBaseWorkChain<{running.pk}>')
        return ToContext(workchain_recon=running)

    def inspect_recon(self):
        """Verify that the reconnaissance PwBaseWorkChain finished successfully."""
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

        u_projection_type_relax = parameters['SYSTEM'].get('u_projection_type', self.defaults.u_projection_type_relax)

        parameters['SYSTEM']['u_projection_type'] = self.defaults.u_projection_type_relax
        inputs.base.pw.parameters = orm.Dict(dict=parameters)
        inputs.metadata.call_link_label = 'iteration_{:02d}_relax'.format(self.ctx.iteration)

        if u_projection_type_relax != self.defaults.u_projection_type_relax:
            self.report(
                f'warning: you specified `u_projection_type = {u_projection_type_relax}` in the input parameters, but '
                r'this will crash pw.x, changing it to `{self.defaults.u_projection_type_relax}`'
            )

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> iteration #{self.ctx.iteration}')
        return ToContext(workchains_relax=append_(running))

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
        workchain = self.ctx.workchains_relax[-1]

        if not workchain.is_finished_ok:
            self.report(f'reconnaissance PwBaseWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX.format(iteration=self.ctx.iteration)

        self.ctx.current_structure = workchain.outputs.output_structure

    def run_scf_fixed(self):
        """Run an scf `PwBaseWorkChain` with fixed occupations."""
        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'
        inputs.pw.parameters['SYSTEM']['occupations'] = 'fixed'
        inputs.pw.parameters['SYSTEM'].pop('degauss', None)
        inputs.pw.parameters['SYSTEM'].pop('smearing', None)
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = inputs.pw.parameters['SYSTEM'].get(
            'u_projection_type', self.defaults.u_projection_type_scf
        )
        conv_thr = inputs.pw.parameters['ELECTRONS'].get('conv_thr', self.defaults.conv_thr_strictfinal)
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = conv_thr
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
        inputs.pw.parameters['SYSTEM']['smearing'] = inputs.pw.parameters['SYSTEM'].get(
            'smearing', self.defaults.smearing_method
        )
        inputs.pw.parameters['SYSTEM']['degauss'] = inputs.pw.parameters['SYSTEM'].get(
            'degauss', self.defaults.smearing_degauss
        )
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = inputs.pw.parameters['SYSTEM'].get(
            'u_projection_type', self.defaults.u_projection_type_scf
        )
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = inputs.pw.parameters['ELECTRONS'].get(
            'conv_thr', self.defaults.conv_thr_preconverge
        )
        inputs.metadata.call_link_label = 'iteration_{:02d}_scf_smearing'.format(self.ctx.iteration)
        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launching PwBaseWorkChain<{running.pk}> with smeared occupations')
        return ToContext(workchains_scf=append_(running))

    def run_scf_fixed_magnetic(self):
        """Run an scf `PwBaseWorkChain` with fixed occupations restarting from the previous calculation.

        The nunmber of bands and total magnetization are set according to those of the previous calculation that was
        run with smeared occupations.
        """
        previous_workchain = self.ctx.workchains_scf[-1]
        previous_parameters = previous_workchain.outputs.output_parameters

        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'
        inputs.pw.parameters['CONTROL']['restart_mode'] = 'restart'
        inputs.pw.parameters['SYSTEM']['occupations'] = 'fixed'
        inputs.pw.parameters['SYSTEM'].pop('degauss', None)
        inputs.pw.parameters['SYSTEM'].pop('smearing', None)
        inputs.pw.parameters['SYSTEM'].pop('starting_magnetization', None)
        inputs.pw.parameters['SYSTEM']['nbnd'] = previous_parameters.get_dict()['number_of_bands']
        inputs.pw.parameters['SYSTEM']['tot_magnetization'] = previous_parameters.get_dict()['total_magnetization']
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = inputs.pw.parameters['SYSTEM'].get(
            'u_projection_type', self.defaults.u_projection_type_scf
        )
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = inputs.pw.parameters['ELECTRONS'].get(
            'conv_thr', self.defaults.conv_thr_strictfinal
        )

        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)
        inputs.metadata.call_link_label = 'iteration_{:02d}_scf_fixed_magnetic'.format(self.ctx.iteration)

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(
            f'launching PwBaseWorkChain<{running.pk}> with fixed occupations, bands and total magnetization'
        )
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

        if not self.inputs.meta_convergence:
            self.report('meta convergence is switched off, so not checking convergence of Hubbard U parameters.')
            self.ctx.is_converged = True
            return

        prev_hubbard_u = self.ctx.current_hubbard_u

        # First check if new types were created, in which case we will have to create a new `StructureData`
        for site in workchain.outputs.hubbard.get_attribute('sites'):
            if site['type'] != site['new_type']:
                self.report('new types have been determined: relabeling the structure and starting new iteration.')
                result = structure_relabel_kinds(self.ctx.current_structure, workchain.outputs.hubbard)
                self.ctx.current_structure = result['structure']
                self.ctx.current_hubbard_u = result['hubbard_u'].get_dict()
                break
        else:
            self.ctx.current_hubbard_u = {}
            for entry in workchain.outputs.hubbard.get_dict()['sites']:
                self.ctx.current_hubbard_u[entry['kind']] = float(entry['value'])

        # Check per site if the new computed value is converged with respect to the last iteration
        for entry in workchain.outputs.hubbard.get_attribute('sites'):
            kind = entry['kind']
            index = entry['index']
            tolerance = self.inputs.tolerance.value
            current_value = float(entry['value'])
            previous_value = float(prev_hubbard_u[kind])
            if abs(current_value - previous_value) > self.inputs.tolerance.value:
                msg = f'parameters not converged for site {index}: {current_value} - {previous_value} > {tolerance}'
                self.report(msg)
                break
        else:
            self.report('Hubbard U parameters are converged')
            self.ctx.is_converged = True

    def run_results(self):
        """Attach the final converged Hubbard U parameters and the corresponding structure."""
        self.report(f'Hubbard U parameters self-consistently converged in {self.ctx.iteration} iterations')
        self.out('structure', self.ctx.current_structure)
        self.out('hubbard', self.ctx.workchains_hp[-1].outputs.hubbard)
