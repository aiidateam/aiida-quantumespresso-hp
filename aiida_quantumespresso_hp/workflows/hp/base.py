# -*- coding: utf-8 -*-
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.base import Bool
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.parsers import ParserFactory
from aiida.work.workchain import while_
from aiida_quantumespresso.common.workchain.base.restart import BaseRestartWorkChain
from aiida_quantumespresso.common.workchain.utils import ErrorHandlerReport
from aiida_quantumespresso.common.workchain.utils import register_error_handler
from aiida_quantumespresso.utils.resources import get_default_options


PwCalculation = CalculationFactory('quantumespresso.pw')
HpCalculation = CalculationFactory('quantumespresso.hp')
HpParser = ParserFactory('quantumespresso.hp')


class HpBaseWorkChain(BaseRestartWorkChain):
    """
    Base workchain to launch a Quantum Espresso hp.x calculation
    """
    _verbose = False
    _calculation_class = HpCalculation

    ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS = 4
    ERROR_MISSING_PERTURBATION_FILE = 5

    @classmethod
    def define(cls, spec):
        super(HpBaseWorkChain, cls).define(spec)
        spec.input('code', valid_type=Code)
        spec.input('qpoints', valid_type=KpointsData)
        spec.input('parent_calculation', valid_type=PwCalculation, required=False)
        spec.input('parent_folder', valid_type=(FolderData, RemoteData), required=False)
        spec.input('parameters', valid_type=ParameterData, required=False)
        spec.input('settings', valid_type=ParameterData, required=False)
        spec.input('options', valid_type=ParameterData, required=False)
        spec.input('only_initialization', valid_type=Bool, default=Bool(False))
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            while_(cls.should_run_calculation)(
                cls.run_calculation,
                cls.inspect_calculation,
            ),
            cls.results,
        )
        spec.output('output_parameters', valid_type=ParameterData)
        spec.output('retrieved', valid_type=FolderData)
        spec.output('matrices', valid_type=ArrayData, required=False)
        spec.output('hubbard', valid_type=ParameterData, required=False)
        spec.output('chi', valid_type=ArrayData, required=False)

    def validate_inputs(self):
        """
        A HpCalculation can be continued either from a completed PwCalculation in which case
        the parent_calculation input should be set, or it can be a restart from a previous HpCalculation
        as for example the final post-processing calculation when parallelizing over atoms and
        or q-points, in which case the parent_folder should be set. In either case, at least one
        of the two inputs has to be defined properly
        """
        self.ctx.inputs = AttributeDict({
            'code': self.inputs.code,
            'qpoints': self.inputs.qpoints,
        })

        if not ('parent_calculation' in self.inputs or 'parent_folder' in self.inputs):
            self.abort_nowait('Neither the parent_calculation nor the parent_folder input was defined')

        try:
            self.ctx.inputs.parent_folder = self.inputs.parent_calculation.out.remote_folder
        except AttributeError:
            self.ctx.inputs.parent_folder = self.inputs.parent_folder

        if 'parameters' in self.inputs:
            self.ctx.inputs.parameters = self.inputs.parameters.get_dict()
        else:
            self.ctx.inputs.parameters = {}

        if 'settings' in self.inputs:
            self.ctx.inputs.settings = self.inputs.settings.get_dict()
        else:
            self.ctx.inputs.settings = {}

        if 'options' in self.inputs:
            self.ctx.inputs.options = self.inputs.options.get_dict()
        else:
            self.ctx.inputs.options = get_default_options()

        if 'INPUTHP'not in self.ctx.inputs.parameters:
            self.ctx.inputs.parameters['INPUTHP'] = {}

        if self.inputs.only_initialization.value:
            self.ctx.inputs.parameters['INPUTHP']['determine_num_pert_only'] = True


@register_error_handler(HpBaseWorkChain, 100)
def _handle_error_incorrect_order_atomic_positions(self, calculation):
    """
    The structure used by the parent calculation has its kinds in the wrong order, as in that the Hubbard kinds
    were not added first, so they will not be printed first in the ATOMIC_POSITIONS card, which is required
    by the hp.x code
    """
    if calculation.finish_status == HpParser.ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS:
        self.report('the parent calculation used a structure where the Hubbard atomic positions did not appear first')
        return ErrorHandlerReport(True, True, self.ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS)


@register_error_handler(HpBaseWorkChain, 200)
def _handle_error_missing_perturbation_file(self, calculation):
    """
    The calculation was run in `collect_chi` mode, however, the code did not find all the perturbation files that
    it expected based on the number of Hubbard kinds in the parent calculation
    """
    if calculation.finish_status == HpParser.ERROR_MISSING_PERTURBATION_FILE:
        self.report('one or more perturbation files that were expected for the collect_chi calculation, are missing')
        return ErrorHandlerReport(True, True, self.ERROR_MISSING_PERTURBATION_FILE)
