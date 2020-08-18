# -*- coding: utf-8 -*-
"""Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the Hubbard atoms."""

from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.engine import WorkChain, ToContext, append_

from aiida_quantumespresso_hp.calculations.functions.collect_atomic_calculations import collect_atomic_calculations

PwCalculation = CalculationFactory('quantumespresso.pw')
HpCalculation = CalculationFactory('quantumespresso.hp')
HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')


class HpParallelizeAtomsWorkChain(WorkChain):
    """Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the Hubbard atoms."""

    ERROR_CHILD_WORKCHAIN_FAILED = 100

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpBaseWorkChain, exclude=('only_initialization',))
        spec.outline(
            cls.run_init,
            cls.run_atoms,
            cls.run_collect,
            cls.run_final,
            cls.results
        )
        spec.expose_outputs(HpBaseWorkChain)
        spec.exit_code(300, 'ERROR_CHILD_WORKCHAIN_FAILED',
            message='A child work chain failed.')
        spec.exit_code(301, 'ERROR_INITIALIZATION_WORKCHAIN_FAILED',
            message='The child work chain failed.')

    def run_init(self):
        """Run an initialization `HpBaseWorkChain` to that will determine which kinds need to be perturbed.

        By performing an `initialization_only` calculation only the symmetry analysis will be performed to determine
        which kinds are to be perturbed. This information is parsed and can be used to determine exactly how many
        `HpBaseWorkChains` have to be launched in parallel.
        """
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.only_initialization = orm.Bool(True)

        running = self.submit(HpBaseWorkChain, **inputs)

        self.report('launched initialization HpBaseWorkChain<{}>'.format(running.pk))

        return ToContext(initialization=running)

    def run_atoms(self):
        """Run a separate `HpBaseWorkChain` for each of the defined Hubbard atoms."""
        workchain = self.ctx.initialization

        if not workchain.is_finished_ok:
            args = (workchain.__class__.__name__, workchain.pk, workchain.exit_status)
            self.report('initialization work chain {}<{}> failed with finish status {}, aborting...'.format(*args))
            return self.exit_codes.ERROR_INITIALIZATION_WORKCHAIN_FAILED

        output_params = workchain.outputs.parameters.get_dict()
        hubbard_sites = output_params['hubbard_sites']

        for site_index, site_kind in hubbard_sites.items():

            do_only_key = 'perturb_only_atom({})'.format(site_index)

            inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
            inputs.hp.parameters = inputs.hp.parameters.get_dict()
            inputs.hp.parameters['INPUTHP'][do_only_key] = True
            inputs.hp.parameters = orm.Dict(dict=inputs.hp.parameters)

            running = self.submit(HpBaseWorkChain, **inputs)

            args = (HpBaseWorkChain.__name__, running.pk, site_index, site_kind)
            self.report('launched {}<{}> for atomic site {} of kind {}'.format(*args))
            self.to_context(workchains=append_(running))

    def run_collect(self):
        """Collect all `retrieved` nodes of the completed `HpBaseWorkChain` and merge them into a single `FolderData`.

        The merged `FolderData` is used as input for a final `HpBaseWorkChain` to compute final matrices.
        """
        retrieved_folders = {}

        for workchain in self.ctx.workchains:

            if not workchain.is_finished_ok:
                args = (workchain.__class__.__name__, workchain.pk, workchain.exit_status)
                self.report('child work chain {}<{}> failed with exit status {}, aborting...'.format(*args))
                return self.exit_codes.ERROR_CHILD_WORKCHAIN_FAILED

            retrieved = workchain.outputs.retrieved
            output_params = workchain.outputs.parameters
            atomic_site_index = list(output_params.get_dict()['hubbard_sites'].keys())[0]
            retrieved_folders['site_index_{}'.format(atomic_site_index)] = retrieved

        self.ctx.merged_retrieved = collect_atomic_calculations(**retrieved_folders)

    def run_final(self):
        """Perform the final HpCalculation to collect the various components of the chi matrices."""
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.hp.parameters = inputs.hp.parameters.get_dict()
        inputs.hp.parameters['INPUTHP']['compute_hp'] = True
        inputs.hp.parameters = orm.Dict(dict=inputs.hp.parameters)
        inputs.hp.parent_folder = self.ctx.merged_retrieved

        running = self.submit(HpBaseWorkChain, **inputs)

        self.report('launched HpBaseWorkChain<{}> to collect matrices'.format(running.pk))
        self.to_context(workchains=append_(running))

    def results(self):
        """Retrieve the results from the final matrix collection workchain."""
        self.out_many(self.exposed_outputs(self.ctx.workchains[-1], HpBaseWorkChain))
