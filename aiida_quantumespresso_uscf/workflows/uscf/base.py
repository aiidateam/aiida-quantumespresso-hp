# -*- coding: utf-8 -*-
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.base import Bool, Int
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.common.datastructures import calc_states
from aiida.work.workchain import WorkChain, ToContext, while_, append_
from aiida.work.run import submit

PwCalculation = CalculationFactory('quantumespresso.pw')
UscfCalculation = CalculationFactory('quantumespresso.uscf')

class UscfBaseWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(UscfBaseWorkChain, cls).define(spec)
        spec.input('code', valid_type=Code)
        spec.input('parent_calculation', valid_type=PwCalculation, required=False)
        spec.input('parent_folder', valid_type=(FolderData, RemoteData), required=False)
        spec.input('qpoints', valid_type=KpointsData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('settings', valid_type=ParameterData)
        spec.input('options', valid_type=ParameterData)
        spec.input('only_initialization', valid_type=Bool, default=Bool(False))
        spec.input('max_iterations', valid_type=Int, default=Int(4))
        spec.outline(
            cls.validate_inputs,
            cls.setup,
            while_(cls.should_run_uscf)(
                cls.run_uscf,
                cls.inspect_uscf,
            ),
            cls.run_results,
        )
        spec.output('parameters', valid_type=ParameterData)
        spec.output('retrieved', valid_type=FolderData)
        spec.output('matrices', valid_type=ArrayData, required=False)
        spec.output('hubbard', valid_type=ParameterData, required=False)
        spec.output('chi', valid_type=ArrayData, required=False)

    def validate_inputs(self):
        """
        A UscfCalculation can be continued either from a completed PwCalculation in which case
        the parent_calculation input should be set, or it can be a restart from a previous UscfCalculation
        as for example the final post-processing calculation when parallelizing over atoms and
        or q-points, in which case the parent_folder should be set. In either case, at least one
        of the two inputs has to be defined properly
        """
        if not ('parent_calculation' in self.inputs or 'parent_folder' in self.inputs):
            self.abort_nowait('Neither the parent_calculation nor the parent_folder input was defined')

        try:
            parent_folder = self.inputs.parent_calculation.out.remote_folder
        except AttributeError:
            parent_folder = self.inputs.parent_folder

        self.ctx.parent_folder = parent_folder

    def setup(self):
        """
        Initialize context variables
        """
        self.ctx.max_iterations = self.inputs.max_iterations.value
        self.ctx.has_calculation_failed = False
        self.ctx.has_submission_failed = False
        self.ctx.is_finished = False
        self.ctx.iteration = 0

        # Define convenience dictionary of inputs for UscfCalculation
        self.ctx.inputs = {
            'code': self.inputs.code,
            'qpoints': self.inputs.qpoints,
            'parameters': self.inputs.parameters,
            'parent_folder': self.ctx.parent_folder,
            'settings': self.inputs.settings,
            '_options': self.inputs.options.get_dict(),
        }

        if self.inputs.only_initialization.value:
            self.ctx.inputs['parameters']['INPUTHP']['determine_num_pert_only'] = True

        return

    def should_run_uscf(self):
        """
        Return whether a restart calculation should be run, which is the case as long as the last
        calculation was not converged successfully and the maximum number of restarts has not yet
        been exceeded
        """
        return not self.ctx.is_finished and self.ctx.iteration < self.ctx.max_iterations

    def run_uscf(self):
        """
        Run the next UscfCalculation
        """
        self.ctx.iteration += 1

        process = UscfCalculation.process()
        running = submit(process, **self.ctx.inputs)

        self.report('launching UscfCalculation<{}> iteration #{}'.format(running.pid, self.ctx.iteration))

        return ToContext(calculations=append_(running))

    def inspect_uscf(self):
        """
        Analyse the results of the previous UscfCalculation, checking whether it finished successfully
        or if not troubleshoot the cause and adapt the input parameters accordingly before
        restarting, or abort if unrecoverable error was found
        """
        try:
            calculation = self.ctx.calculations[-1]
        except Exception:
            self.abort_nowait('the first iteration finished without returning a UscfCalculation')
            return

        expected_states = [calc_states.FINISHED, calc_states.FAILED, calc_states.SUBMISSIONFAILED]

        # Done: successful convergence of last calculation
        if calculation.has_finished_ok():
            self.report('converged successfully after {} iterations'.format(self.ctx.iteration))
            self.ctx.parent_folder = calculation.out.remote_folder
            self.ctx.is_finished = True

        # Abort: exceeded maximum number of retries
        elif self.ctx.iteration >= self.ctx.max_iterations:
            self.report('reached the maximum number of iterations {}'.format(self.ctx.max_iterations))
            self.abort_nowait('last ran UscfCalculation<{}>'.format(calculation.pk))

        # Abort: unexpected state of last calculation
        elif calculation.get_state() not in expected_states:
            self.abort_nowait('unexpected state ({}) of UscfCalculation<{}>'.format(
                calculation.get_state(), calculation.pk))

        # Retry: submission failed, try to restart or abort
        elif calculation.get_state() in [calc_states.SUBMISSIONFAILED]:
            self._handle_submission_failure(calculation)
            self.ctx.has_submission_failed = True

        # Retry: calculation failed, try to salvage or abort
        elif calculation.get_state() in [calc_states.FAILED]:
            self._handle_calculation_failure(calculation)
            self.ctx.has_submission_failed = False

        return

    def run_results(self):
        """
        Attach the output parameters and retrieved folder of the last calculation to the outputs
        """
        calculation = self.ctx.calculations[-1]

        # Only non-parallelized or matrix collecting calculations will have all output links
        for link in ['retrieved', 'parameters', 'chi', 'hubbard', 'matrices']:
            if link in calculation.out:
                self.out(link, calculation.out[link])

        self.report('workchain completed after {} iterations'.format(self.ctx.iteration))

    def _handle_submission_failure(self, calculation):
        """
        The submission of the calculation has failed, if it was the second consecutive failure we
        abort the workchain, else we set the has_submission_failed flag and try again
        """
        if self.ctx.has_submission_failed:
            self.abort_nowait('submission for UscfCalculation<{}> failed for the second time'.format(
                calculation.pk))
        else:
            self.report('submission for UscfCalculation<{}> failed, retrying once more'.format(
                calculation.pk))

    def _handle_calculation_failure(self, calculation):
        """
        The calculation has failed so we try to analyze the reason and change the inputs accordingly
        """
        if 'not_converged' in calculation.res.parser_warnings:
            self.ctx.has_calculation_failed = False
            params = self.ctx.inputs['parameters'].get_dict()
            params['INPUTHP']['niter_ph'] = 100
            self.ctx.inputs['parameters'] = ParameterData(dict=params)
            self.report('UscfCalculation<{}> did not converge, restarting'.format(calculation.pk))

        else:
            self._handle_unexpected_calculation_failure(calculation)

    def _handle_unexpected_calculation_failure(self, calculation):
        """
        The calculation has failed unexpectedly. If it was the first time, simply try to restart.
        If it was the second unexpected failure, report the warnings and parser warnings and abort
        """
        if self.ctx.has_calculation_failed:
            warnings = '\n'.join([w.strip() for w in calculation.res.warnings])
            parser_warnings = '\n'.join([w.strip() for w in calculation.res.parser_warnings])
            self.report('UscfCalculation<{}> failed unexpectedly'.format(calculation.pk))
            self.report('list of warnings: {}'.format(warnings))
            self.report('list of parser warnings: {}'.format(parser_warnings))
            self.abort_nowait('second unexpected failure in a row'.format(calculation.pk))
        else:
            self.ctx.has_calculation_failed = True
            self.report('UscfCalculation<{}> failed unexpectedly, restarting once more'.format(calculation.pk))