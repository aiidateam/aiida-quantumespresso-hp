# -*- coding: utf-8 -*-
from aiida.orm import Code, CalculationFactory, WorkflowFactory
from aiida.orm.data.base import Bool, Int
from aiida.orm.data.array import ArrayData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.work.workchain import WorkChain, ToContext


PwCalculation = CalculationFactory('quantumespresso.pw')
HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')
HpParallelizeAtomsWorkChain = WorkflowFactory('quantumespresso.hp.parallelize_atoms')


class HpWorkChain(WorkChain):
    """
    Workchain that will run a Quantum Espresso hp.x calculation based on a previously completed
    PwCalculation. If specified through the 'parallelize_atoms' boolean input parameter, the
    calculation will be parallelized over the Hubbard atoms by running the HpParallelizeAtomsWorkChain.
    Otherwise a single HpBaseWorkChain will be launched that will compute every Hubbard atom serially.
    """

    @classmethod
    def define(cls, spec):
        super(HpWorkChain, cls).define(spec)
        spec.input('code', valid_type=Code)
        spec.input('parent_calculation', valid_type=PwCalculation)
        spec.input('qpoints', valid_type=KpointsData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('settings', valid_type=ParameterData)
        spec.input('options', valid_type=ParameterData)
        spec.input('max_iterations', valid_type=Int, default=Int(10))
        spec.input('parallelize_atoms', valid_type=Bool, default=Bool(False))
        spec.outline(
            cls.run_workchain,
            cls.run_results,
        )
        spec.output('parameters', valid_type=ParameterData)
        spec.output('retrieved', valid_type=FolderData)
        spec.output('matrices', valid_type=ArrayData)
        spec.output('hubbard', valid_type=ParameterData)
        spec.output('chi', valid_type=ArrayData)

    def run_workchain(self):
        """
        If parallelize_atoms is true, run the HpParallelizeAtomsWorkChain, otherwise run HpBaseWorkChain
        """
        inputs = {
            'code': self.inputs.code,
            'parent_calculation': self.inputs.parent_calculation,
            'qpoints': self.inputs.qpoints,
            'parameters': self.inputs.parameters,
            'settings': self.inputs.settings,
            'options': self.inputs.options,
            'max_iterations': self.inputs.max_iterations,
        }

        if self.inputs.parallelize_atoms:
            running = self.submit(HpParallelizeAtomsWorkChain, **inputs)

            self.report('running in parallel, launching HpParallelizeAtomsWorkChain<{}>'.format(running.pk))
            return ToContext(workchain=running)
        else:
            running = self.submit(HpBaseWorkChain, **inputs)

            self.report('running in serial, launching HpBaseWorkChain<{}>'.format(running.pk))
            return ToContext(workchain=running)

    def run_results(self):
        """
        Retrieve the results from the completed sub workchain
        """
        workchain = self.ctx.workchain

        for link in ['retrieved', 'parameters', 'chi', 'hubbard', 'matrices']:
            if not link in workchain.out:
                self.abort_nowait("the sub workchain is missing expected output link '{}'".format(link))
            else:
                self.out(link, workchain.out[link])

        self.report('workchain completed successfully')