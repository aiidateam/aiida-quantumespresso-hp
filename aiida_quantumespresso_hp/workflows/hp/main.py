# -*- coding: utf-8 -*-
from aiida.work.run import submit
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.base import Bool, Int, Str
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.work.workchain import WorkChain, ToContext
from aiida_quantumespresso_hp.workflows.hp.base import HpBaseWorkChain
from aiida_quantumespresso_hp.workflows.hp.parallelize_atoms import HpParallelizeAtomsWorkChain

PwCalculation = CalculationFactory('quantumespresso.pw')

class HpWorkChain(WorkChain):
    """
    Workchain that will run a Quantum Espresso Hp.x calculation based on a previously completed
    PwCalculation. If specified through the 'parallelize_atoms' boolean input parameter, the
    calculation will be parallelized over the Hubbard atoms by running the HpParallelizeAtomsWorkChain.
    Otherwise a single HpBaseWorkChain will be launched that will compute every Hubbard atom serially.
    """
    def __init__(self, *args, **kwargs):
        super(HpWorkChain, self).__init__(*args, **kwargs)

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
        spec.dynamic_output()

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
            running = submit(HpParallelizeAtomsWorkChain, **inputs)

            self.report('running in parallel, launching HpParallelizeAtomsWorkChain<{}>'.format(running.pid))
            return ToContext(workchain=running)
        else:
            running = submit(HpBaseWorkChain, **inputs)

            self.report('running in serial, launching HpBaseWorkChain<{}>'.format(running.pid))
            return ToContext(workchain=running)

    def run_results(self):
        """
        Documentation string
        """
        output_hubbard = self.ctx.workchain.out.hubbard
        self.out('output_hubbard', output_hubbard)

        self.report('workchain completed, output in {}<{}>'.format(type(output_hubbard), output_hubbard.pk))

        return