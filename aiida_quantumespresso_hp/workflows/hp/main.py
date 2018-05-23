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

    ERROR_CHILD_WORKCHAIN_FAILED = 100

    @classmethod
    def define(cls, spec):
        super(HpWorkChain, cls).define(spec)
        spec.expose_inputs(HpBaseWorkChain)
        spec.input('parallelize_atoms', valid_type=Bool, default=Bool(False))
        spec.outline(
            cls.run_workchain,
            cls.inspect_workchain,
            cls.results,
        )
        spec.expose_outputs(HpBaseWorkChain)

    def run_workchain(self):
        """
        If parallelize_atoms is true, run the HpParallelizeAtomsWorkChain, otherwise run HpBaseWorkChain
        """
        if self.inputs.parallelize_atoms:
            running = self.submit(HpParallelizeAtomsWorkChain, **self.exposed_inputs(HpBaseWorkChain))
            self.report('running in parallel, launching HpParallelizeAtomsWorkChain<{}>'.format(running.pk))
            return ToContext(workchain=running)
        else:
            running = self.submit(HpBaseWorkChain, **self.exposed_inputs(HpBaseWorkChain))
            self.report('running in serial, launching HpBaseWorkChain<{}>'.format(running.pk))
            return ToContext(workchain=running)

    def inspect_workchain(self):
        """
        Verify that the child workchain has finished successfully
        """
        if not self.ctx.workchain.is_finished_ok:
            self.report('the {} workchain did not finish successfully'.format(self.ctx.workchain.process_label))
            return self.ERROR_CHILD_WORKCHAIN_FAILED

    def results(self):
        """
        Retrieve the results from the completed sub workchain
        """
        self.out_many(self.exposed_outputs(self.ctx.workchain, HpBaseWorkChain))
