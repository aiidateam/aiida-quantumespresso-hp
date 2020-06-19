# -*- coding: utf-8 -*-
"""Work chain to run a Quantum ESPRESSO hp.x calculation."""
from aiida import orm
from aiida.engine import WorkChain, ToContext, if_
from aiida.plugins import WorkflowFactory

HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')
HpParallelizeAtomsWorkChain = WorkflowFactory('quantumespresso.hp.parallelize_atoms')


class HpWorkChain(WorkChain):
    """Work chain to run a Quantum ESPRESSO hp.x calculation.

    If the `parallelize_atoms` input is set to `True`, the calculation will be parallelized over the Hubbard atoms by
    running the `HpParallelizeAtomsWorkChain`. Otherwise a single `HpBaseWorkChain` will be launched that will compute
    every Hubbard atom in serial.
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpBaseWorkChain, exclude=('only_initialization',))
        spec.input('parallelize_atoms', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.outline(
            if_(cls.should_parallelize_atoms)(
                cls.run_parallel_workchain,
            ).else_(
                cls.run_base_workchain,
            ),
            cls.inspect_workchain,
            cls.results,
        )
        spec.expose_outputs(HpBaseWorkChain)
        spec.exit_code(300, 'ERROR_CHILD_WORKCHAIN_FAILED', message='A child work chain failed.')

    def should_parallelize_atoms(self):
        """Return whether the calculation should be parallelized over atoms."""
        return self.inputs.parallelize_atoms.value

    def run_base_workchain(self):
        """Run the `HpBaseWorkChain`."""
        running = self.submit(HpBaseWorkChain, **self.exposed_inputs(HpBaseWorkChain))
        self.report('running in serial, launching HpBaseWorkChain<{}>'.format(running.pk))
        return ToContext(workchain=running)

    def run_parallel_workchain(self):
        """Run the `HpParallelizeAtomsWorkChain`."""
        running = self.submit(HpParallelizeAtomsWorkChain, **self.exposed_inputs(HpBaseWorkChain))
        self.report('running in parallel, launching HpParallelizeAtomsWorkChain<{}>'.format(running.pk))
        return ToContext(workchain=running)

    def inspect_workchain(self):
        """Verify that the child workchain has finished successfully."""
        if not self.ctx.workchain.is_finished_ok:
            self.report('the {} workchain did not finish successfully'.format(self.ctx.workchain.process_label))
            return self.exit_codes.ERROR_CHILD_WORKCHAIN_FAILED

    def results(self):
        """Retrieve the results from the completed sub workchain."""
        self.out_many(self.exposed_outputs(self.ctx.workchain, HpBaseWorkChain))
