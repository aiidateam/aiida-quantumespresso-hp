# -*- coding: utf-8 -*-
"""Work chain to run a Quantum ESPRESSO hp.x calculation."""
from aiida import orm
from aiida.engine import WorkChain, ToContext, if_
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

KpointsData = DataFactory('array.kpoints')
HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')
HpParallelizeAtomsWorkChain = WorkflowFactory('quantumespresso.hp.parallelize_atoms')


class HpWorkChain(ProtocolMixin, WorkChain):
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

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        parent_scf_folder=None,
        protocol=None,
        overrides=None,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        qpoints = KpointsData()
        qpoints.set_kpoints_mesh(inputs['qpoints_mesh'])

        builder = cls.get_builder()
        builder.hp.code = code
        builder.hp.parameters = orm.Dict(dict=inputs['hp']['parameters'])
        if parent_scf_folder is not None:
            builder.hp.parent_scf = parent_scf_folder
        builder.hp.qpoints = qpoints
        builder.parallelize_atoms = orm.Bool(inputs['parallelize_atoms'])

        return builder

    def should_parallelize_atoms(self):
        """Return whether the calculation should be parallelized over atoms."""
        return self.inputs.parallelize_atoms.value

    def run_base_workchain(self):
        """Run the `HpBaseWorkChain`."""
        running = self.submit(HpBaseWorkChain, **self.exposed_inputs(HpBaseWorkChain))
        self.report(f'running in serial, launching HpBaseWorkChain<{running.pk}>')
        return ToContext(workchain=running)

    def run_parallel_workchain(self):
        """Run the `HpParallelizeAtomsWorkChain`."""
        running = self.submit(HpParallelizeAtomsWorkChain, **self.exposed_inputs(HpBaseWorkChain))
        self.report(f'running in parallel, launching HpParallelizeAtomsWorkChain<{running.pk}>')
        return ToContext(workchain=running)

    def inspect_workchain(self):
        """Verify that the child workchain has finished successfully."""
        if not self.ctx.workchain.is_finished_ok:
            self.report(f'the {self.ctx.workchain.process_label} workchain did not finish successfully')
            return self.exit_codes.ERROR_CHILD_WORKCHAIN_FAILED

    def results(self):
        """Retrieve the results from the completed sub workchain."""
        self.out_many(self.exposed_outputs(self.ctx.workchain, HpBaseWorkChain))
