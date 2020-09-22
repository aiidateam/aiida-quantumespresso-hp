# -*- coding: utf-8 -*-
"""Turn-key solution to automatically compute the self-consistent Hubbard parameters for a given structure."""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
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


class SelfConsistentHubbardWorkChain(WorkChain):
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
            self.report('reordered StructureData<{}>'.format(structure.pk))

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

    def run_recon(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation.

        This runs a simple scf cycle with a few steps with smearing turned on to determine whether the system is most
        likely a metal or an insulator. This step is required because the metallicity of the systems determines how the
        relaxation calculations in the convergence cycle have to be performed.
        """
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='recon'))
        parameters = inputs.pw.parameters.get_dict()

        parameters.setdefault('CONTROL', {})['calculation'] = 'scf'
        parameters.setdefault('ELECTRONS', {})['scf_must_converge'] = False
        parameters.setdefault('ELECTRONS', {})['electron_maxstep'] = 10
        parameters.setdefault('SYSTEM', {})['occupations'] = 'smearing'
        parameters.setdefault('SYSTEM', {})['smearing'] = self.defaults.smearing_method
        parameters.setdefault('SYSTEM', {})['degauss'] = self.defaults.smearing_degauss
        parameters.setdefault('SYSTEM', {}).pop('lda_plus_u', None)

        inputs.pw.parameters = orm.Dict(dict=parameters)
        inputs.pw.structure = self.ctx.current_structure

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching reconnaissance PwBaseWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_recon=running)

    def inspect_recon(self):
        """Verify that the reconnaissance PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchain_recon

        if not workchain.is_finished_ok:
            self.report('reconnaissance PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
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
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        parameters = inputs.base.pw.parameters.get_dict()

        parameters.setdefault('SYSTEM', {})
        u_projection_type_relax = parameters['SYSTEM'].get('u_projection_type', self.defaults.u_projection_type_relax)

        parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
        parameters['SYSTEM']['u_projection_type'] = self.defaults.u_projection_type_relax

        if u_projection_type_relax != self.defaults.u_projection_type_relax:
            self.report(
                "warning: you specified 'u_projection_type = {}' in the input parameters, but this will crash "
                "pw.x, changing it to '{}'".format(u_projection_type_relax, self.defaults.u_projection_type_relax)
            )

        inputs.base.pw.parameters = orm.Dict(dict=parameters)
        inputs.structure = self.ctx.current_structure

        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report('launching PwRelaxWorkChain<{}> iteration #{}'.format(running.pk, self.ctx.iteration))

        return ToContext(workchains_relax=append_(running))

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
        workchain = self.ctx.workchains_relax[-1]

        if not workchain.is_finished_ok:
            self.report('reconnaissance PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX.format(iteration=self.ctx.iteration)

        self.ctx.current_structure = workchain.outputs.output_structure

    def run_scf_fixed(self):
        """Run an scf `PwBaseWorkChain` with fixed occupations."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict()

        parameters.setdefault('CONTROL', {})
        parameters.setdefault('SYSTEM', {})
        parameters.setdefault('ELECTRONS', {})

        parameters['CONTROL']['calculation'] = 'scf'
        parameters['SYSTEM']['occupations'] = 'fixed'
        parameters['SYSTEM'].pop('degauss', None)
        parameters['SYSTEM'].pop('smearing', None)
        parameters['SYSTEM']['u_projection_type'] = parameters['SYSTEM'].get(
            'u_projection_type', self.defaults.u_projection_type_scf
        )
        conv_thr = parameters['ELECTRONS'].get('conv_thr', self.defaults.conv_thr_strictfinal)
        parameters['ELECTRONS']['conv_thr'] = conv_thr

        parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
        inputs.pw.parameters = orm.Dict(dict=parameters)
        inputs.pw.structure = self.ctx.current_structure

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> with fixed occupations'.format(running.pk))

        return ToContext(workchains_scf=append_(running))

    def run_scf_smearing(self):
        """Run an scf `PwBaseWorkChain` with smeared occupations"""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict()

        parameters.setdefault('CONTROL', {})
        parameters.setdefault('SYSTEM', {})
        parameters.setdefault('ELECTRONS', {})

        parameters['CONTROL']['calculation'] = 'scf'
        parameters['SYSTEM']['occupations'] = 'smearing'
        parameters['SYSTEM']['smearing'] = parameters['SYSTEM'].get(
            'smearing', self.defaults.smearing_method
        )
        parameters['SYSTEM']['degauss'] = parameters['SYSTEM'].get(
            'degauss', self.defaults.smearing_degauss
        )
        parameters['SYSTEM']['u_projection_type'] = parameters['SYSTEM'].get(
            'u_projection_type', self.defaults.u_projection_type_scf
        )
        parameters['ELECTRONS']['conv_thr'] = parameters['ELECTRONS'].get(
            'conv_thr', self.defaults.conv_thr_preconverge
        )

        parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
        inputs.pw.parameters = orm.Dict(dict=parameters)
        inputs.pw.structure = self.ctx.current_structure

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> with smeared occupations'.format(running.pk))

        return ToContext(workchains_scf=append_(running))

    def run_scf_fixed_magnetic(self):
        """Run an scf `PwBaseWorkChain` with fixed occupations restarting from the previous calculation.

        The nunmber of bands and total magnetization are set according to those of the previous calculation that was
        run with smeared occupations.
        """
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        parameters = inputs.pw.parameters.get_dict()

        previous_workchain = self.ctx.workchains_scf[-1]
        previous_parameters = previous_workchain.out.output_parameters

        parameters.setdefault('CONTROL', {})
        parameters.setdefault('SYSTEM', {})
        parameters.setdefault('ELECTRONS', {})

        parameters['CONTROL']['calculation'] = 'scf'
        parameters['CONTROL']['restart_mode'] = 'restart'
        parameters['SYSTEM']['occupations'] = 'fixed'
        parameters['SYSTEM'].pop('degauss', None)
        parameters['SYSTEM'].pop('smearing', None)
        parameters['SYSTEM'].pop('starting_magnetization', None)
        parameters['SYSTEM']['nbnd'] = previous_parameters.get_dict()['number_of_bands']
        parameters['SYSTEM']['tot_magnetization'] = previous_parameters.get_dict()['total_magnetization']
        parameters['SYSTEM']['u_projection_type'] = parameters['SYSTEM'].get(
            'u_projection_type', self.defaults.u_projection_type_scf
        )
        parameters['ELECTRONS']['conv_thr'] = inputs.parameters['ELECTRONS'].get(
            'conv_thr', self.defaults.conv_thr_strictfinal
        )

        parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
        inputs.pw.parameters = orm.Dict(dict=parameters)
        inputs.pw.parameters = orm.Dict(dict=parameters)

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(
            'launching PwBaseWorkChain<{}> with fixed occupations, bands and total magnetization'.format(running.pk)
        )

        return ToContext(workchains_scf=append_(running))

    def inspect_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchains_scf[-1]

        if not workchain.is_finished_ok:
            args = [self.ctx.iteration, workchain.exit_status]
            self.report('scf PwBaseWorkChain in iteration {} failed with exit status {}'.format(*args))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF.format(iteration=self.ctx.iteration)

    def run_hp(self):
        """
        Run the HpWorkChain restarting from the last completed scf calculation
        """
        workchain = self.ctx.workchains_scf[-1]

        inputs = AttributeDict(self.exposed_inputs(HpWorkChain, namespace='hubbard'))
        inputs.hp.parent_scf = workchain.outputs.remote_folder

        running = self.submit(HpWorkChain, **inputs)

        self.report('launching HpWorkChain<{}> iteration #{}'.format(running.pk, self.ctx.iteration))

        return ToContext(workchains_hp=append_(running))

    def inspect_hp(self):
        """
        Analyze the last completed HpWorkChain. We check the current Hubbard U parameters and compare those with
        the values computed in the previous iteration. If the difference for all Hubbard sites is smaller than
        the tolerance, the calculation is considered to be converged.
        """
        workchain = self.ctx.workchains_hp[-1]

        if not workchain.is_finished_ok:
            args = [self.ctx.iteration, workchain.exit_status]
            self.report('HpWorkChain in iteration {} failed with exit status {}'.format(*args))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_HP.format(iteration=self.ctx.iteration)

        hubbard = workchain.outputs.hubbard

        prev_hubbard_u = self.ctx.current_hubbard_u
        curr_hubbard_u = {}

        for entry in hubbard.get_dict()['sites']:
            curr_hubbard_u[entry['kind']] = float(entry['value'])

        # Compare new Hubbard U with values from previous iteration to check the convergence
        converged = True
        for kind in curr_hubbard_u:
            prev_value = prev_hubbard_u[kind]
            curr_value = curr_hubbard_u[kind]
            if abs(curr_value - prev_value) > self.inputs.tolerance.value:
                converged = False

        self.ctx.current_hubbard_u = curr_hubbard_u
        self.ctx.is_converged = converged

        if not self.inputs.meta_convergence:
            self.ctx.is_converged = True
        else:
            if converged:
                self.report('Hubbard U parameters are converged')
            else:
                self.report('Hubbard U parameters are not converged')

            self.report(
                'values from previous iteration: {}'.format(' '.join([str(v) for v in prev_hubbard_u.values()]))
            )
            self.report('values from current iteration: {}'.format(' '.join([str(v) for v in curr_hubbard_u.values()])))

        return

    def run_results(self):
        """Attach the final converged Hubbard U parameters and the corresponding structure."""
        self.report('Hubbard U parameters self-consistently converged in {} iterations'.format(self.ctx.iteration))
        self.out('structure', self.ctx.current_structure)
        self.out('hubbard', self.ctx.workchains_hp[-1].outputs.hubbard)
