# -*- coding: utf-8 -*-
"""Workchain to run a Quantum ESPRESSO hp.x calculation with automated error handling and restarts."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import BaseRestartWorkChain, ProcessHandlerReport, process_handler, while_
from aiida.plugins import CalculationFactory

HpCalculation = CalculationFactory('quantumespresso.hp')


class HpBaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a Quantum ESPRESSO hp.x calculation with automated error handling and restarts."""

    _process_class = HpCalculation

    defaults = AttributeDict({
        'delta_factor_alpha_mix': 0.5,
        'delta_factor_niter_max': 2,
    })

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpCalculation, namespace='hp')
        spec.input('only_initialization', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.outline(
            cls.setup,
            cls.validate_parameters,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.expose_outputs(HpCalculation)
        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unrecoverable error.')

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

    @process_handler(priority=500, exit_codes=HpCalculation.exit_codes.ERROR_CONVERGENCE_NOT_REACHED)
    def handle_convergence_not_reached(self, _):
        """Handle `ERROR_CONVERGENCE_NOT_REACHED`: decrease `alpha_mix`, increase `niter_max`, and restart.

        Since `hp.x` does not support restarting from incomplete calculations, the entire calculation will have to be
        restarted from scratch. By increasing the `niter_max` and decreasing `alpha_mix` there is a chance that the next
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

        if 'niter_max' in parameters:
            parameters['niter_max'] *= self.defaults.delta_factor_niter_max
        else:
            parameters['niter_max'] = 200

        changes.append(f"changed `niter_max` to {parameters['niter_max']}")

        if changes:
            self.report(f"convergence not reached: {', '.join(changes)}")
        else:
            self.report('convergence not reached, restarting')

        return ProcessHandlerReport(True)