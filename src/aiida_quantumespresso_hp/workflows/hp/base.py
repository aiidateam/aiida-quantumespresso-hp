# -*- coding: utf-8 -*-
"""Workchain to run a Quantum ESPRESSO hp.x calculation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, process_handler, while_
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

HpCalculation = CalculationFactory('quantumespresso.hp')


class HpBaseWorkChain(BaseRestartWorkChain, ProtocolMixin):
    """Workchain to run a Quantum ESPRESSO hp.x calculation with automated error handling and restarts."""

    _process_class = HpCalculation

    defaults = AttributeDict({
        'delta_factor_alpha_mix': 0.5,
        'delta_factor_niter_max': 2,
        'delta_factor_max_seconds': 0.95,
    })

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpCalculation, namespace='hp')
        spec.input('only_initialization', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.outline(
            cls.setup,
            cls.validate_parameters,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.expose_outputs(HpCalculation)
        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unrecoverable error.')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ..protocols import hp as hp_protocols
        return files(hp_protocols) / 'base.yaml'

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        protocol=None,
        parent_scf_folder=None,
        parent_hp_folders: dict = None,
        overrides=None,
        options=None,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.hp`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param parent_scf_folder: the parent ``RemoteData`` of the respective SCF calcualtion.
        :param parent_hp_folders: the parent ``FolderData`` of the respective single atoms HP calcualtions.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

        if isinstance(code, str):
            code = orm.load_code(code)

        type_check(code, orm.AbstractCode)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        # Update the parameters based on the protocol inputs
        parameters = inputs['hp']['parameters']
        metadata = inputs['hp']['metadata']

        qpoints_mesh = inputs['hp'].pop('qpoints')
        qpoints = orm.KpointsData()
        qpoints.set_kpoints_mesh(qpoints_mesh)

        # If overrides are provided, they are considered absolute
        if overrides:
            parameter_overrides = overrides.get('hp', {}).get('parameters', {})
            parameters = recursive_merge(parameters, parameter_overrides)

        if options:
            metadata['options'] = recursive_merge(inputs['hp']['metadata']['options'], options)

        hubbard_structure = inputs['hp'].pop('hubbard_structure', None)
        parent_scf = parent_scf_folder if not 'parent_scf' in inputs['hp'] else inputs['hp']['parent_scf']
        parent_hp = parent_hp_folders if not 'parent_scf' in inputs['hp'] else inputs['hp']['parent_scf']

        # pylint: disable=no-member
        builder = cls.get_builder()
        builder.hp['code'] = code
        builder.hp['qpoints'] = qpoints
        builder.hp['parameters'] = orm.Dict(parameters)
        builder.hp['metadata'] = metadata
        if 'settings' in inputs['hp']:
            builder.hp['settings'] = orm.Dict(inputs['hp']['settings'])
        if hubbard_structure:
            builder.hp['hubbard_structure'] = hubbard_structure
        if parent_scf:
            builder.hp['parent_scf'] = parent_scf
        if parent_hp:
            builder.hp['parent_hp'] = parent_hp
        builder.only_initialization = orm.Bool(inputs['only_initialization'])
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])
        # pylint: enable=no-member

        return builder

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super().setup()
        self.ctx.restart_calc = None
        self.ctx.inputs = AttributeDict(self.exposed_inputs(HpCalculation, 'hp'))

    def validate_parameters(self):
        """Validate inputs that might depend on each other and cannot be validated by the spec."""
        self.ctx.inputs.parameters = self.ctx.inputs.parameters.get_dict()
        self.ctx.inputs.parameters.setdefault('INPUTHP', {})

        if self.inputs.only_initialization.value:
            self.ctx.inputs.parameters['INPUTHP']['determine_num_pert_only'] = True

        self.ctx.inputs.settings = self.ctx.inputs.settings.get_dict() if 'settings' in self.ctx.inputs else {}

    def set_max_seconds(self, max_wallclock_seconds):
        """Set the `max_seconds` to a fraction of `max_wallclock_seconds` option to prevent out-of-walltime problems.

        :param max_wallclock_seconds: the maximum wallclock time that will be set in the scheduler settings.
        """
        max_seconds_factor = self.defaults.delta_factor_max_seconds
        max_seconds = max_wallclock_seconds * max_seconds_factor
        self.ctx.inputs.parameters['INPUTHP']['max_seconds'] = max_seconds

    def prepare_process(self):
        """Prepare the inputs for the next calculation."""
        max_wallclock_seconds = self.ctx.inputs.metadata.options.get('max_wallclock_seconds', None)

        if max_wallclock_seconds is not None and 'max_seconds' not in self.ctx.inputs.parameters['INPUTHP']:
            self.set_max_seconds(max_wallclock_seconds)

    def report_error_handled(self, node, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param node: the failed calculation node
        :param action: a string message with the action taken
        """
        self.report(f'{node.process_label}<{node.pk}> failed with exit status {node.exit_status}: {node.exit_message}')
        self.report(f'Action taken: {action}')

    @process_handler(priority=600)
    def handle_unrecoverable_failure(self, node):
        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
        if node.is_failed and node.exit_status < 400:
            self.report_error_handled(node, 'unrecoverable error, aborting...')
            return ProcessHandlerReport(True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

    @process_handler(priority=460, exit_codes=HpCalculation.exit_codes.ERROR_COMPUTING_CHOLESKY)
    def handle_computing_cholesky(self, _):
        """Handle `ERROR_COMPUTING_CHOLESKY`: set parallel diagonalization to 1 and restart.

        Parallelization of diagonalization may produce in some cases too much numerical noise,
        giving rise to Cholesky factorization issues. As other diagonalization algorithms are
        not available in `hp.x`, we try to set the diagonalization flag to 1, if not already set.
        """
        settings = self.ctx.inputs.settings
        cmdline = settings.get('cmdline', [])

        for key in ['-ndiag', '-northo', '-nd']:
            if key in cmdline:

                index = cmdline.index(key)

                if int(cmdline[index+1]) == 1:
                    self.report('diagonalization flag already to 1, stopping')
                    return ProcessHandlerReport(False)

                cmdline[index+1] = '1' # enforce to be 1
                break
        else:
            cmdline += ['-nd', '1']

        settings['cmdline'] = cmdline
        self.report('set parallelization flag for diagonalization to 1, restarting')
        return ProcessHandlerReport(True)


    @process_handler(priority=410, exit_codes=HpCalculation.exit_codes.ERROR_CONVERGENCE_NOT_REACHED)
    def handle_convergence_not_reached(self, _):
        """Handle `ERROR_CONVERGENCE_NOT_REACHED`: decrease `alpha_mix` and restart.

        Since `hp.x` does not support restarting from incomplete calculations, the entire calculation will have to be
        restarted from scratch. By decreasing `alpha_mix` there is a chance that the next
        run will converge. If these keys are present in the input parameters, they will be scaled by a default factor,
        otherwise, a hardcoded default value will be set that is lower/higher than that of the code's default.
        """
        parameters = self.ctx.inputs.parameters['INPUTHP']
        changes = []

        # The `alpha_mix` parameter is an array and so all keys matching `alpha_mix(i)` with `i` some integer should
        # be corrected accordingly. If no such key exists, the default `alpha_mix(1)` is set.
        if any(parameter.startswith('alpha_mix(') for parameter in parameters.keys()):
            for parameter in parameters.keys():
                if parameter.startswith('alpha_mix('):
                    parameters[parameter] *= self.defaults.delta_factor_alpha_mix
                    changes.append(f'changed `{parameter}` to {parameters[parameter]}')
        else:
            parameter = 'alpha_mix(1)'
            parameters[parameter] = 0.20
            changes.append(f'set `{parameter}` to {parameters[parameter]}')

        if changes:
            self.report(f"convergence not reached: {', '.join(changes)}")
        else:
            self.report('convergence not reached, restarting')

        return ProcessHandlerReport(True)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

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
