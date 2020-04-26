# -*- coding: utf-8 -*-
"""Workchain to run a Quantum ESPRESSO hp.x calculation with automated error handling and restarts."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import while_
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.common.workchain.base.restart import BaseRestartWorkChain
from aiida_quantumespresso.common.workchain.utils import ErrorHandlerReport, register_error_handler

HpCalculation = CalculationFactory('quantumespresso.hp')


class HpBaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a Quantum ESPRESSO hp.x calculation with automated error handling and restarts."""

    _calculation_class = HpCalculation

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpCalculation, namespace='hp')
        spec.input('only_initialization', valid_type=orm.Bool, default=orm.Bool(False))
        spec.outline(
            cls.setup,
            cls.validate_parameters,
            cls.validate_resources,
            while_(cls.should_run_calculation)(
                cls.run_calculation,
                cls.inspect_calculation,
            ),
            cls.results,
        )
        spec.exit_code(204, 'ERROR_INVALID_INPUT_RESOURCES_UNDERSPECIFIED',
            message='The `metadata.options` did not specify both `resources.num_machines` and `max_wallclock_seconds`.')
        spec.exit_code(300, 'ERROR_UNRECOVERABLE_FAILURE',
            message='The calculation failed with an unrecoverable error.')
        spec.expose_outputs(HpCalculation)

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(HpCalculation, 'hp'))

    def validate_parameters(self):
        """Validate inputs that might depend on each other and cannot be validated by the spec."""
        self.ctx.inputs.parameters = self.ctx.inputs.parameters.get_dict()
        self.ctx.inputs.parameters.setdefault('INPUTHP', {})

        if self.inputs.only_initialization.value:
            self.ctx.inputs.parameters['INPUTHP']['determine_num_pert_only'] = True

    def validate_resources(self):
        """Validate the inputs related to the resources.

        The `metadata.options` should at least contain the options `resources` and `max_wallclock_seconds`, where
        `resources` should define the `num_machines`.
        """
        num_machines = self.ctx.inputs.metadata.options.get('resources', {}).get('num_machines', None)
        max_wallclock_seconds = self.ctx.inputs.metadata.options.get('max_wallclock_seconds', None)

        if num_machines is None or max_wallclock_seconds is None:
            return self.exit_codes.ERROR_INVALID_INPUT_RESOURCES_UNDERSPECIFIED

        # self.set_max_seconds(max_wallclock_seconds)

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report('Action taken: {}'.format(action))


@register_error_handler(HpBaseWorkChain, 600)
def _handle_unrecoverable_failure(self, calculation):
    """Calculations with an exit status below 400 are unrecoverable, so abort the work chain."""
    if calculation.exit_status < 400:
        self.report_error_handled(calculation, 'unrecoverable error, aborting...')
        return ErrorHandlerReport(True, True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)
