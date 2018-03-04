# -*- coding: utf-8 -*-
from copy import deepcopy
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.base import Bool
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.work.workchain import while_
from aiida_quantumespresso.utils.resources import get_default_options
from aiida_quantumespresso.common.workchain.base.restart import BaseRestartWorkChain


PwCalculation = CalculationFactory('quantumespresso.pw')
HpCalculation = CalculationFactory('quantumespresso.hp')


class HpBaseWorkChain(BaseRestartWorkChain):
    """
    Base workchain to launch a Quantum Espresso hp.x calculation
    """
    _verbose = False
    _calculation_class = HpCalculation

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
        spec.output('parameters', valid_type=ParameterData)
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
        self.ctx.inputs_raw = AttributeDict({
            'code': self.inputs.code,
            'qpoints': self.inputs.qpoints,
        })

        if not ('parent_calculation' in self.inputs or 'parent_folder' in self.inputs):
            self.abort_nowait('Neither the parent_calculation nor the parent_folder input was defined')

        try:
            self.ctx.inputs_raw.parent_folder = self.inputs.parent_calculation.out.remote_folder
        except AttributeError:
            self.ctx.inputs_raw.parent_folder = self.inputs.parent_folder

        if 'parameters' in self.inputs:
            self.ctx.inputs_raw.parameters = self.inputs.parameters.get_dict()
        else:
            self.ctx.inputs_raw.parameters = {}

        if 'settings' in self.inputs:
            self.ctx.inputs_raw.settings = self.inputs.settings.get_dict()
        else:
            self.ctx.inputs_raw.settings = {}

        if 'options' in self.inputs:
            self.ctx.inputs_raw.options = self.inputs.options.get_dict()
        else:
            self.ctx.inputs_raw.options = get_default_options()

        if 'INPUTHP'not in self.ctx.inputs_raw.parameters:
            self.ctx.inputs_raw.parameters['INPUTHP'] = {}

        if self.inputs.only_initialization.value:
            self.ctx.inputs_raw.parameters['INPUTHP']['determine_num_pert_only'] = True

        # Assign a deepcopy to self.ctx.inputs which will be used by the BaseRestartWorkChain
        self.ctx.inputs = deepcopy(self.ctx.inputs_raw)