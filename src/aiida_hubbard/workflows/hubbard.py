# -*- coding: utf-8 -*-
"""Turn-key solution to automatically compute the self-consistent Hubbard parameters for a given structure."""
from __future__ import annotations

from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
from aiida_quantumespresso.utils.hubbard import HubbardUtils
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
import numpy as np

from aiida_hubbard.calculations.functions.structure_relabel_kinds import structure_relabel_kinds
from aiida_hubbard.calculations.functions.structure_reorder_kinds import structure_reorder_kinds
from aiida_hubbard.utils.general import set_tot_magnetization

HubbardStructureData = DataFactory('quantumespresso.hubbard_structure')

PwCalculation = CalculationFactory('quantumespresso.pw')

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
HpWorkChain = WorkflowFactory('quantumespresso.hp.main')


def get_separated_parameters(
    hubbard_parameters: list[tuple[int, str, int, str, float, tuple[int, int, int], str]]
) -> tuple[list, list]:
    """Return a tuple with onsites and intersites parameters separated.

    :return: tuple (list of onsites, list of intersites).
    """
    onsites = []
    intersites = []

    for parameters in hubbard_parameters:
        if parameters[0] == parameters[2] and parameters[1] == parameters[3]:
            onsites.append(parameters)
        else:
            intersites.append(parameters)

    return onsites, intersites


def validate_positive(value, _):
    """Validate that the value is positive."""
    if value.value < 0:
        return 'the value must be positive.'


def validate_inputs(inputs, _):
    """Validate the entire inputs."""
    parameters = AttributeDict(inputs).scf.pw.parameters.get_dict()
    nspin = parameters.get('SYSTEM', {}).get('nspin', 1)

    if nspin == 2:
        magnetic_moments = parameters.get('SYSTEM', {}).get('starting_magnetization', None)
        if magnetic_moments is None:
            return 'Missing `starting_magnetization` input in `scf.pw.parameters` while `nspin == 2`.'

    if nspin not in [1, 2]:
        return f'nspin=`{nspin}` is not implemented in the `hp.x` code.'


class SelfConsistentHubbardWorkChain(WorkChain, ProtocolMixin):
    """Workchain computing the self-consistent Hubbard parameters of a structure.

    It iteratively relaxes the structure (optional) with the ``PwRelaxWorkChain``
    and computes the Hubbard parameters through the ``HpWorkChain``,
    using the remote folder of an scf performed via the ``PwBaseWorkChain``,
    until the Hubbard values are converged within certain tolerance(s).

    The procedure in each step of the convergence cycle is slightly different depending on the electronic and
    magnetic properties of the system. Each cycle will roughly consist of three steps:

        * Relaxing the structure at the current Hubbard values (optional).
        * One or two DFT calculations depending whether the system is metallic or insulating, respectively.
        * A DFPT calculation of the Hubbard parameters, perturbing the ground-state of the last DFT run.

    The possible options for the set of DFT SCF calculations that have to be run in the second step look are:

        * Metals:
            - SCF with smearing.
        * Insulators
            - SCF with smearing.
            - SCF with fixed occupations; if magnetic, total magnetization and number of bands
                are fixed to the values found from the previous SCF calculation.

    When convergence is achieved a node will be returned containing the final converged
    :class:`~aiida_quantumespresso.data.hubbard_structure.HubbardStructureData`.
    """

    defaults = AttributeDict({
        'qe': qe_defaults,
        'smearing_method': 'cold',
        'smearing_degauss': 0.01,
        'conv_thr_preconverge': 1E-10,
        'conv_thr_strictfinal': 1E-15,
    })

    @classmethod
    def define(cls, spec):
        """Define the specifications of the process."""
        # yapf: disable
        super().define(spec)
        spec.input('hubbard_structure', valid_type=HubbardStructureData,
            help=('The HubbardStructureData containing the initialized parameters for triggering '
                  'the Hubbard atoms which the `hp.x` code will perturbe.'))
        spec.input('tolerance_onsite', valid_type=orm.Float, default=lambda: orm.Float(0.1),
            help=('Tolerance value for self-consistent calculation of Hubbard U. '
                  'In case of DFT+U+V calculation, it refers to the diagonal elements (i.e. on-site).'))
        spec.input('tolerance_intersite', valid_type=orm.Float, default=lambda: orm.Float(0.01),
            help=('Tolerance value for self-consistent DFT+U+V calculation. '
                  'It refers to the only off-diagonal elements V.'))
        spec.input('skip_relax_iterations', valid_type=orm.Int, required=False, validator=validate_positive,
            help=('The number of iterations for skipping the `relax` '
                  'step without performing check on parameters convergence.'))
        spec.input('radial_analysis', valid_type=orm.Dict, required=False,
            help='If specified, it performs a nearest neighbour analysis and feed the radius to hp.x')
        spec.input('relax_frequency', valid_type=orm.Int, required=False, validator=validate_positive,
            help='Integer value referring to the number of iterations to wait before performing the `relax` step.')
        spec.expose_inputs(PwRelaxWorkChain, namespace='relax',
            exclude=('clean_workdir', 'structure', 'base_final_scf'),
            namespace_options={'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain` that, when defined, will iteratively relax the structure.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir','pw.structure'))
        spec.expose_inputs(HpWorkChain, namespace='hubbard',
            exclude=('clean_workdir', 'hp.parent_scf', 'hp.parent_hp', 'hp.hubbard_structure'))
        spec.input('max_iterations', valid_type=orm.Int, default=lambda: orm.Int(10),
            help='Maximum number of iterations of the (relax-)scf-hp cycle.')
        spec.input('meta_convergence', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='Whether performing the self-consistent cycle. If False, it will stop at the first iteration.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.inputs.validator = validate_inputs
        spec.inputs['hubbard']['hp'].validator = None

        spec.outline(
            cls.setup,
            while_(cls.should_run_iteration)(
                cls.update_iteration,
                if_(cls.should_run_relax)(
                    cls.run_relax,
                    cls.inspect_relax,
                ),
                cls.run_scf_smearing,
                cls.recon_scf,
                if_(cls.is_insulator)(
                    cls.run_scf_fixed,
                    cls.inspect_scf,
                ),
                cls.run_hp,
                cls.inspect_hp,
                if_(cls.should_check_convergence)(
                    cls.check_convergence,
                ),
                if_(cls.should_clean_workdir)(
                    cls.clean_iteration,
                ),
            ),
            cls.run_results,
        )

        spec.output('hubbard_structure', valid_type=HubbardStructureData, required=False,
            help='The Hubbard structure containing the structure and associated Hubbard parameters.')

        spec.exit_code(330, 'ERROR_FAILED_TO_DETERMINE_PSEUDO_POTENTIAL',
            message='Failed to determine the correct pseudo potential after the structure changed its kind names.')
        spec.exit_code(340, 'ERROR_RELABELLING_KINDS',
            message='Failed to determine the kind names during the relabelling.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RECON',
            message='The reconnaissance PwBaseWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='The PwRelaxWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The scf PwBaseWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_HP',
            message='The HpWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(405, 'ERROR_NON_INTEGER_TOT_MAGNETIZATION',
            message='The scf PwBaseWorkChain sub process in iteration {iteration}'\
                    'returned a non integer total magnetization (threshold exceeded).')

        spec.exit_code(601, 'ERROR_CONVERGENCE_NOT_REACHED',
            message='The Hubbard parameters did not converge at the last iteration #{iteration}')
        # yapf: enable

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols
        return files(protocols) / 'hubbard.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        hp_code,
        hubbard_structure,
        protocol=None,
        overrides=None,
        options_pw=None,
        options_hp=None,
        **kwargs
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param hp_code: the ``Code`` instance configured for the ``quantumespresso.hp`` plugin.
        :param hubbard_structure: the ``HubbardStructureData`` instance containing the initialised Hubbard paramters.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options_pw: A dictionary of options that will be recursively set for the ``metadata.options``
            input of all the pw ``CalcJobs`` that are nested in this work chain.
        :param options_hp: A dictionary of options that will be recursively set for the ``metadata.options``
            input of all the hp ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        args = (pw_code, hubbard_structure, protocol)
        relax = PwRelaxWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('relax', None), options=options_pw, **kwargs
        )
        scf = PwBaseWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('scf', None), options=options_pw, **kwargs
        )

        args = (hp_code, protocol)
        hubbard = HpWorkChain.get_builder_from_protocol(
            *args, overrides=inputs.get('hubbard', None), options=options_hp, **kwargs
        )

        relax.pop('clean_workdir')
        relax.pop('structure')
        relax.pop('base_final_scf', None)  # We do not want to run a final scf, since it would be time wasted.
        scf.pop('clean_workdir')
        scf['pw'].pop('structure')

        hubbard.pop('clean_workdir', None)
        for namespace in ('parent_scf', 'hubbard_structure', 'parent_hp'):
            hubbard['hp'].pop(namespace, None)

        builder = cls.get_builder()

        if 'relax_frequency' in inputs:
            builder.relax_frequency = orm.Int(inputs['relax_frequency'])
        if 'skip_relax_iterations' in inputs:
            builder.skip_relax_iterations = orm.Int(inputs['skip_relax_iterations'])

        builder.hubbard_structure = hubbard_structure
        builder.relax = relax
        builder.scf = scf
        builder.hubbard = hubbard
        builder.tolerance_onsite = orm.Float(inputs['tolerance_onsite'])
        builder.tolerance_intersite = orm.Float(inputs['tolerance_intersite'])
        builder.radial_analysis = orm.Dict(inputs['radial_analysis'])
        builder.max_iterations = orm.Int(inputs['max_iterations'])
        builder.meta_convergence = orm.Bool(inputs['meta_convergence'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def setup(self):
        """Set up Context variables."""
        # Set ctx variables for the cycle.
        self.ctx.current_magnetic_moments = None  # starting_magnetization dict for collinear spin calcs
        self.ctx.max_iterations = self.inputs.max_iterations.value
        self.ctx.is_converged = False
        self.ctx.is_insulator = None
        self.ctx.is_magnetic = False
        self.ctx.iteration = 0
        self.ctx.skip_relax_iterations = 0
        if 'skip_relax_iterations' in self.inputs:
            self.ctx.skip_relax_iterations = self.inputs.skip_relax_iterations.value
        self.ctx.relax_frequency = 1
        if 'relax_frequency' in self.inputs:
            self.ctx.relax_frequency = self.inputs.relax_frequency.value

        # Check if the atoms should be reordered
        hp_utils = HubbardUtils(self.inputs.hubbard_structure)
        if not hp_utils.is_to_reorder():
            self.ctx.current_hubbard_structure = self.inputs.hubbard_structure
        else:
            self.report('detected kinds in the wrong order: reordering the kinds.')
            self.ctx.current_hubbard_structure = structure_reorder_kinds(self.inputs.hubbard_structure)

        # Determine whether the system is to be treated as magnetic
        parameters = self.inputs.scf.pw.parameters.get_dict()
        nspin = parameters.get('SYSTEM', {}).get('nspin', self.defaults.qe.nspin)
        magnetic_moments = parameters.get('SYSTEM', {}).get('starting_magnetization', None)

        if nspin == 1:
            self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
        else:
            self.report('system is treated to be magnetic because `nspin != 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = True
            self.ctx.current_magnetic_moments = orm.Dict(magnetic_moments)

    def should_run_relax(self):
        """Return whether a relax calculation needs to be run, which is true if `relax` is specified in inputs."""
        if 'relax' not in self.inputs:
            return False

        if self.ctx.iteration <= self.ctx.skip_relax_iterations:
            self.report((
                f'`skip_relax_iterations` is set to {self.ctx.skip_relax_iterations}. '
                f'Skipping relaxation for iteration {self.ctx.iteration}.'
            ))
            return False

        if self.ctx.iteration % self.ctx.relax_frequency != 0:
            self.report((
                f'`relax_frequency` is set to {self.ctx.relax_frequency}. '
                f'Skipping relaxation for iteration {self.ctx.iteration}.'
            ))
            return False

        return True

    def should_check_convergence(self):
        """Return whether to check the convergence of Hubbard parameters."""
        if not self.inputs.meta_convergence.value:
            return False

        if self.ctx.iteration <= self.ctx.skip_relax_iterations:
            self.report((
                f'`skip_relax_iterations` is set to {self.ctx.skip_relax_iterations}. '
                f'Skipping convergence check for iteration {self.ctx.iteration}.'
            ))
            return False

        return True

    def should_run_iteration(self):
        """Return whether a new process should be run."""
        return not self.ctx.is_converged and self.ctx.iteration < self.ctx.max_iterations

    def update_iteration(self):
        """Update the current iteration index counter."""
        self.ctx.iteration += 1

    def is_insulator(self):
        """Return whether the current structure is a metal."""
        return self.ctx.is_insulator

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

        if cls is PwBaseWorkChain and namespace == 'scf':
            inputs = self.set_pw_parameters(inputs)
            inputs.pw.pseudos = pseudos
            inputs.pw.structure = self.ctx.current_hubbard_structure

        elif cls is PwRelaxWorkChain and namespace == 'relax':
            inputs.base = self.set_pw_parameters(inputs.base)
            # inputs.base.pw.parameters.setdefault('IONS', {})
            inputs.structure = self.ctx.current_hubbard_structure
            inputs.base.pw.pseudos = pseudos
            inputs.pop('base_final_scf', None)  # We do not want to run a final scf, since it would be time wasted.

        return inputs

    def set_pw_parameters(self, inputs):
        """Set the input parameters for a generic `quantumespresso.pw` calculation.

        :param inputs: AttributeDict of a ``PwBaseWorkChain`` builder input.
        """
        parameters = inputs.pw.parameters.get_dict()
        parameters.setdefault('CONTROL', {})
        parameters.setdefault('SYSTEM', {})
        parameters.setdefault('ELECTRONS', {})

        if self.ctx.current_magnetic_moments:
            parameters['SYSTEM']['starting_magnetization'] = self.ctx.current_magnetic_moments.get_dict()

        inputs.pw.parameters = orm.Dict(parameters)

        return inputs

    def get_pseudos(self) -> dict:
        """Return the mapping of pseudos based on the current structure.

        .. note:: this is necessary because during the workchain the kind names of the structure can change, meaning the
            mapping of the pseudos that is to be passed to the subprocesses also may have to change, since the keys are
            based on the kind names of the structure.

        :return: dictionary of pseudos where the keys are the kindnames of ``self.ctx.current_hubbard_structure``.
        """
        import re

        results = {}
        pseudos = self.inputs.scf.pw.pseudos

        for kind in self.ctx.current_hubbard_structure.kinds:
            for key, pseudo in pseudos.items():
                symbol = re.sub(r'\d', '', key)
                if re.match(fr'{kind.symbol}[0-9]*', symbol):
                    results[kind.name] = pseudo
                    break
            else:
                raise ValueError(f'could not find the pseudo from inputs.scf.pw.pseudos for kind `{kind}`.')

        return results

    def relabel_hubbard_structure(self, workchain) -> None:
        """Relabel the Hubbard structure if new types have been detected."""
        from aiida_quantumespresso.utils.hubbard import is_intersite_hubbard

        if not is_intersite_hubbard(workchain.outputs.hubbard_structure.hubbard):
            for site in workchain.outputs.hubbard.dict.sites:
                if not site['type'] == site['new_type']:
                    try:
                        result = structure_relabel_kinds(
                            self.ctx.current_hubbard_structure, workchain.outputs.hubbard,
                            self.ctx.current_magnetic_moments
                        )
                    except ValueError:
                        return self.exit_codes.ERROR_RELABELLING_KINDS
                    self.ctx.current_hubbard_structure = result['hubbard_structure']
                    if self.ctx.current_magnetic_moments is not None:
                        self.ctx.current_magnetic_moments = result['starting_magnetization']
                    self.report('new types have been detected: relabeling the structure.')
                    return

    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation."""
        inputs = self.get_inputs(PwRelaxWorkChain, 'relax')
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.metadata.call_link_label = f'iteration_{self.ctx.iteration:02d}_relax'

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> iteration #{self.ctx.iteration}')
        return ToContext(workchains_relax=append_(running))

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
        workchain = self.ctx.workchains_relax[-1]

        if not workchain.is_finished_ok:
            self.report(f'PwRelaxWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX.format(iteration=self.ctx.iteration)

        self.ctx.current_hubbard_structure = workchain.outputs.output_structure

    def run_scf_smearing(self):
        """Run an scf `PwBaseWorkChain` with smeared occupations.

        This step is always needed since we do not a priori whether
        the material will be metallic or insulating.
        """
        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        parameters = inputs.pw.parameters
        parameters['CONTROL']['calculation'] = 'scf'
        parameters['SYSTEM']['occupations'] = 'smearing'
        parameters['SYSTEM']['smearing'] = parameters['SYSTEM'].get('smearing', self.defaults.smearing_method)
        parameters['SYSTEM']['degauss'] = parameters['SYSTEM'].get('degauss', self.defaults.smearing_degauss)
        parameters['ELECTRONS']['conv_thr'] = parameters['ELECTRONS'].get(
            'conv_thr', self.defaults.conv_thr_preconverge
        )
        inputs.pw.parameters = orm.Dict(parameters)
        inputs.metadata.call_link_label = f'iteration_{self.ctx.iteration:02d}_scf_smearing'

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launching PwBaseWorkChain<{running.pk}> with smeared occupations')
        return ToContext(workchains_scf=append_(running))

    def run_scf_fixed(self):
        """Run an scf `PwBaseWorkChain` with fixed occupations on top of the previous calculation.

        The nunmber of bands and total magnetization (if magnetic) are set according to those of the
        previous calculation that was run with smeared occupations.

        .. note: this will be run only if the material has been recognised as insulating.
        """
        previous_workchain = self.ctx.workchains_scf[-1]
        previous_parameters = previous_workchain.outputs.output_parameters

        inputs = self.get_inputs(PwBaseWorkChain, 'scf')

        nbnd = previous_parameters.get_dict()['number_of_bands']
        conv_thr = inputs.pw.parameters['ELECTRONS'].get('conv_thr', self.defaults.conv_thr_strictfinal)

        inputs.pw.parameters['CONTROL'].update({
            'calculation': 'scf',
            'restart_mode': 'from_scratch',  # important
        })
        inputs.pw.parameters['SYSTEM'].update({
            'nbnd': nbnd,
            'occupations': 'fixed',
        })
        inputs.pw.parameters['ELECTRONS'].update({
            'conv_thr': conv_thr,
            'startingpot': 'file',
            'startingwfc': 'file',
        })

        for key in ['degauss', 'smearing', 'starting_magnetization']:
            inputs.pw.parameters['SYSTEM'].pop(key, None)

        # If magnetic, set the total magnetization and raises an error if is non (not close enough) integer.
        if self.ctx.is_magnetic:
            total_magnetization = previous_parameters.get_dict()['total_magnetization']
            if not set_tot_magnetization(inputs.pw.parameters, total_magnetization):
                return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION.format(iteration=self.ctx.iteration)

        inputs.pw.parent_folder = previous_workchain.outputs.remote_folder
        inputs.pw.parameters = orm.Dict(inputs.pw.parameters)

        if self.ctx.is_magnetic:
            inputs.metadata.call_link_label = f'iteration_{self.ctx.iteration:02d}_scf_fixed_magnetic'
            report_append = 'bands and total magnetization'
        else:
            inputs.metadata.call_link_label = f'iteration_{self.ctx.iteration:02d}_scf_fixed'
            report_append = ''

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching PwBaseWorkChain<{running.pk}> with fixed occupations' + report_append)

        return ToContext(workchains_scf=append_(running))

    def inspect_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchains_scf[-1]

        if not workchain.is_finished_ok:
            self.report(f'scf in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF.format(iteration=self.ctx.iteration)

    def recon_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchains_scf[-1]

        if not workchain.is_finished_ok:
            self.report(f'scf in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF.format(iteration=self.ctx.iteration)

        bands = workchain.outputs.output_band
        parameters = workchain.outputs.output_parameters.get_dict()

        fermi_energy = parameters['fermi_energy']
        number_electrons = parameters['number_of_electrons']

        # Due to uncertainty in the prediction of the fermi energy, we try
        # both options of this function. If one of the two give an insulating
        # state as a result, we then set fixed occupation as it is likely that
        # hp.x would crash otherwise.
        is_insulator_1, _ = find_bandgap(bands, fermi_energy=fermi_energy)

        # I am not sure, but I think for some materials, e.g. having anti-ferromagnetic
        # ordering, the following function would crash for some reason, possibly due
        # to the format of the BandsData. To double check if actually needed.
        try:
            is_insulator_2, _ = find_bandgap(bands, number_electrons=number_electrons)
        except:  # pylint: disable=bare-except
            is_insulator_2 = False

        if is_insulator_1 or is_insulator_2:
            self.report('after relaxation, system is determined to be an insulator')
            self.ctx.is_insulator = True
        else:
            self.report('after relaxation, system is determined to be a metal')
            self.ctx.is_insulator = False

    def run_hp(self):
        """Run the HpWorkChain restarting from the last completed scf calculation."""
        workchain = self.ctx.workchains_scf[-1]

        inputs = AttributeDict(self.exposed_inputs(HpWorkChain, namespace='hubbard'))

        if 'radial_analysis' in self.inputs:
            kwargs = self.inputs.radial_analysis.get_dict()
            hubbard_utils = HubbardUtils(self.ctx.current_hubbard_structure)
            num_neigh = hubbard_utils.get_max_number_of_neighbours(**kwargs)

            parameters = inputs.hp.parameters.get_dict()
            parameters['INPUTHP']['num_neigh'] = num_neigh

            settings = {'radial_analysis': self.inputs.radial_analysis.get_dict()}
            if 'settings' in inputs.hp:
                settings = inputs.hp.settings.get_dict()
                settings['radial_analysis'] = self.inputs.radial_analysis.get_dict()

            inputs.hp.parameters = orm.Dict(parameters)
            inputs.hp.settings = orm.Dict(settings)

        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.hp.parent_scf = workchain.outputs.remote_folder
        inputs.hp.hubbard_structure = self.ctx.current_hubbard_structure
        inputs.metadata.call_link_label = f'iteration_{self.ctx.iteration:02d}_hp'

        running = self.submit(HpWorkChain, **inputs)

        self.report(f'launching HpWorkChain<{running.pk}> iteration #{self.ctx.iteration}')
        self.to_context(**{'workchains_hp': append_(running)})

    def inspect_hp(self):
        """Analyze the last completed HpWorkChain.

        We check the current Hubbard parameters and compare those with the values computed
        in the previous iteration. If the difference for all Hubbard sites is smaller than
        the tolerance(s), the calculation is considered to be converged.
        """
        workchain = self.ctx.workchains_hp[-1]

        if not workchain.is_finished_ok:
            self.report(f'hp.x in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_HP.format(iteration=self.ctx.iteration)

        if not self.should_check_convergence():
            self.ctx.current_hubbard_structure = workchain.outputs.hubbard_structure
            self.relabel_hubbard_structure(workchain)

            if not self.inputs.meta_convergence:
                self.report('meta convergence is switched off, so not checking convergence of Hubbard parameters.')
                self.ctx.is_converged = True

    def check_convergence(self):
        """Check the convergence of the Hubbard parameters."""
        workchain = self.ctx.workchains_hp[-1]

        # We store in memory the parameters before relabelling to make the comparison easier.
        reference = self.ctx.current_hubbard_structure.clone()
        ref_utils = HubbardUtils(reference)
        ref_utils.reorder_atoms()
        ref_params = reference.hubbard.to_list()

        new_hubbard_structure = workchain.outputs.hubbard_structure.clone()
        new_utils = HubbardUtils(reference)
        new_utils.reorder_atoms()
        new_params = new_hubbard_structure.hubbard.to_list()

        # We check if new types were created, in which case we relabel the `HubbardStructureData`
        self.ctx.current_hubbard_structure = workchain.outputs.hubbard_structure
        self.relabel_hubbard_structure(workchain)

        # if not self.should_check_convergence():
        #     return

        if not len(ref_params) == len(new_params):
            self.report('The new and old Hubbard parameters have different lenghts. Assuming to be at the first cycle.')
            return

        ref_onsites, ref_intersites = get_separated_parameters(ref_params)
        new_onsites, new_intersites = get_separated_parameters(new_params)

        check_onsites = True
        check_intersites = True

        # We do the check on the onsites first
        old = np.array(ref_onsites, dtype='object')
        new = np.array(new_onsites, dtype='object')
        diff = np.abs(old[:, 4] - new[:, 4])

        if (diff > self.inputs.tolerance_onsite).any():
            check_onsites = False
            self.report(f'Hubbard onsites parameters are not converged. Max difference is {diff.max()}.')

        # Then the intersites if present. It might be an "only U" calculation.
        if ref_intersites:
            old = np.array(ref_intersites, dtype='object')
            new = np.array(new_intersites, dtype='object')
            diff = np.abs(old[:, 4] - new[:, 4])

            if (diff > self.inputs.tolerance_intersite).any():
                check_onsites = False
                self.report(f'Hubbard intersites parameters are not converged. Max difference is {diff.max()}.')

        if check_intersites and check_onsites:
            self.report('Hubbard parameters are converged. Stopping the cycle.')
            self.ctx.is_converged = True

    def run_results(self):
        """Attach the final converged Hubbard U parameters and the corresponding structure."""
        self.out('hubbard_structure', self.ctx.current_hubbard_structure)

        if self.ctx.is_converged:
            self.report(f'Hubbard parameters self-consistently converged in {self.ctx.iteration} iterations')
        else:
            self.report(f'Hubbard parameters did not converged at the last iteration #{self.ctx.iteration}.')
            return self.exit_codes.ERROR_CONVERGENCE_NOT_REACHED.format(iteration=self.ctx.iteration)

    def should_clean_workdir(self):
        """Whether to clean the work directories at each iteration."""
        return self.inputs.clean_workdir.value

    def clean_iteration(self):
        """Clean all work directiories of the current iteration."""
        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
