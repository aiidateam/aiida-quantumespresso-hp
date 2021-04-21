# -*- coding: utf-8 -*-
"""Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the Hubbard atoms."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida.plugins import DataFactory, CalculationFactory, WorkflowFactory

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

KpointsData = DataFactory('array.kpoints')
PwCalculation = CalculationFactory('quantumespresso.pw')
HpCalculation = CalculationFactory('quantumespresso.hp')
HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')


class HpParallelizeAtomsWorkChain(WorkChain):
    """Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the Hubbard atoms."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpBaseWorkChain, exclude=('only_initialization',))
        spec.outline(
            cls.run_init,
            cls.run_atoms,
            cls.run_final,
            cls.results
        )
        spec.expose_outputs(HpBaseWorkChain)
        spec.exit_code(300, 'ERROR_CHILD_WORKCHAIN_FAILED',
            message='A child work chain failed.')
        spec.exit_code(301, 'ERROR_INITIALIZATION_WORKCHAIN_FAILED',
            message='The child work chain failed.')

    @classmethod
    def get_builder_from_protocol(
        cls,
        code,
        parent_scf_folder,
        protocol=None,
        overrides=None,
        **_
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol."""
        inputs = cls.get_protocol_inputs(protocol, overrides)

        qpoints = KpointsData()
        qpoints.set_kpoints_mesh(inputs['qpoints_mesh'])

        builder = cls.get_builder()
        builder.hp.code = code
        builder.hp.parameters = orm.Dict(dict=inputs['hp']['parameters'])
        builder.hp.parent_scf = parent_scf_folder
        builder.hp.qpoints = qpoints

        return builder

    def run_init(self):
        """Run an initialization `HpBaseWorkChain` to that will determine which kinds need to be perturbed.

        By performing an `initialization_only` calculation only the symmetry analysis will be performed to determine
        which kinds are to be perturbed. This information is parsed and can be used to determine exactly how many
        `HpBaseWorkChains` have to be launched in parallel.
        """
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.only_initialization = orm.Bool(True)
        inputs.metadata.call_link_label = 'initialization'

        node = self.submit(HpBaseWorkChain, **inputs)
        self.to_context(initialization=node)
        self.report(f'launched initialization HpBaseWorkChain<{node.pk}>')

    def run_atoms(self):
        """Run a separate `HpBaseWorkChain` for each of the defined Hubbard atoms."""
        workchain = self.ctx.initialization

        if not workchain.is_finished_ok:
            self.report(f'initialization work chain {workchain} failed with status {workchain.exit_status}, aborting.')
            return self.exit_codes.ERROR_INITIALIZATION_WORKCHAIN_FAILED

        output_params = workchain.outputs.parameters.get_dict()
        hubbard_sites = output_params['hubbard_sites']

        for site_index, site_kind in hubbard_sites.items():

            do_only_key = f'perturb_only_atom({site_index})'
            key = f'atom_{site_index}'

            inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
            inputs.hp.parameters = inputs.hp.parameters.get_dict()
            inputs.hp.parameters['INPUTHP'][do_only_key] = True
            inputs.hp.parameters = orm.Dict(dict=inputs.hp.parameters)
            inputs.metadata.call_link_label = key

            node = self.submit(HpBaseWorkChain, **inputs)
            self.to_context(**{key: node})
            name = HpBaseWorkChain.__name__
            self.report(f'launched {name}<{node.pk}> for atomic site {site_index} of kind {site_kind}')

    def run_final(self):
        """Perform the final HpCalculation to collect the various components of the chi matrices."""
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.hp.parent_scf = inputs.hp.parent_scf
        inputs.hp.parent_hp = {key: wc.outputs.retrieved for key, wc in self.ctx.items() if key.startswith('atom_')}
        inputs.metadata.call_link_label = 'compute_hp'

        node = self.submit(HpBaseWorkChain, **inputs)
        self.to_context(compute_hp=node)
        self.report(f'launched HpBaseWorkChain<{node.pk}> to collect matrices')

    def results(self):
        """Retrieve the results from the final matrix collection workchain."""
        self.out_many(self.exposed_outputs(self.ctx.compute_hp, HpBaseWorkChain))
