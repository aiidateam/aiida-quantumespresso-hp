# -*- coding: utf-8 -*-
"""Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the Hubbard atoms."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain
from aiida.plugins import CalculationFactory, WorkflowFactory

PwCalculation = CalculationFactory('quantumespresso.pw')
HpCalculation = CalculationFactory('quantumespresso.hp')
HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')
HpParallelizeQpointsWorkChain = WorkflowFactory('quantumespresso.hp.parallelize_qpoints')


class HpParallelizeAtomsWorkChain(WorkChain):
    """Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the Hubbard atoms."""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpBaseWorkChain, exclude=('only_initialization', 'clean_workdir'))
        spec.input('parallelize_qpoints', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.run_init,
            cls.inspect_init,
            cls.run_atoms,
            cls.inspect_atoms,
            cls.run_final,
            cls.inspect_final,
            cls.results
        )
        spec.expose_outputs(HpBaseWorkChain)
        spec.exit_code(300, 'ERROR_ATOM_WORKCHAIN_FAILED',
            message='A child work chain failed.')
        spec.exit_code(301, 'ERROR_INITIALIZATION_WORKCHAIN_FAILED',
            message='The child work chain failed.')
        spec.exit_code(302, 'ERROR_FINAL_WORKCHAIN_FAILED',
            message='The child work chain failed.')


    def run_init(self):
        """Run an initialization `HpBaseWorkChain` to that will determine which kinds need to be perturbed.

        By performing an `initialization_only` calculation only the symmetry analysis will be performed to determine
        which kinds are to be perturbed. This information is parsed and can be used to determine exactly how many
        `HpBaseWorkChains` have to be launched in parallel.
        """
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.only_initialization = orm.Bool(True)
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.hp.metadata.options.max_wallclock_seconds = 3600 # 1 hour is more than enough
        inputs.metadata.call_link_label = 'initialization'

        node = self.submit(HpBaseWorkChain, **inputs)
        self.to_context(initialization=node)
        self.report(f'launched initialization HpBaseWorkChain<{node.pk}>')

    def inspect_init(self):
        """Inspect the initialization `HpBaseWorkChain`."""
        workchain = self.ctx.initialization

        if not workchain.is_finished_ok:
            self.report(f'initialization work chain {workchain} failed with status {workchain.exit_status}, aborting.')
            return self.exit_codes.ERROR_INITIALIZATION_WORKCHAIN_FAILED

    def run_atoms(self):
        """Run a separate `HpBaseWorkChain` for each of the defined Hubbard atoms."""
        workchain = self.ctx.initialization

        output_params = workchain.outputs.parameters.get_dict()
        hubbard_sites = output_params['hubbard_sites']

        parallelize_qpoints = self.inputs.parallelize_qpoints.value
        workflow = HpParallelizeQpointsWorkChain if parallelize_qpoints else HpBaseWorkChain

        for site_index, site_kind in hubbard_sites.items():

            do_only_key = f'perturb_only_atom({site_index})'
            key = f'atom_{site_index}'

            inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
            inputs.clean_workdir = self.inputs.clean_workdir
            inputs.hp.parameters = inputs.hp.parameters.get_dict()
            inputs.hp.parameters['INPUTHP'][do_only_key] = True
            inputs.hp.parameters = orm.Dict(dict=inputs.hp.parameters)
            inputs.metadata.call_link_label = key

            node = self.submit(workflow, **inputs)
            self.to_context(**{key: node})
            name = workflow.__name__
            self.report(f'launched {name}<{node.pk}> for atomic site {site_index} of kind {site_kind}')

    def inspect_atoms(self):
        """Inspect each parallel atom `HpBaseWorkChain`."""
        for key, workchain in self.ctx.items():
            if key.startswith('atom_'):
                if not workchain.is_finished_ok:
                    self.report(f'child work chain {workchain} failed with status {workchain.exit_status}, aborting.')
                    return self.exit_codes.ERROR_ATOM_WORKCHAIN_FAILED

    def run_final(self):
        """Perform the final `HpCalculation` to collect the various components of the chi matrices."""
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.hp.parent_scf = inputs.hp.parent_scf
        inputs.hp.parent_hp = {key: wc.outputs.retrieved for key, wc in self.ctx.items() if key.startswith('atom_')}
        inputs.hp.metadata.options.max_wallclock_seconds =  3600 # 1 hour is more than enough
        inputs.metadata.call_link_label = 'compute_hp'

        node = self.submit(HpBaseWorkChain, **inputs)
        self.to_context(compute_hp=node)
        self.report(f'launched HpBaseWorkChain<{node.pk}> to collect matrices')

    def inspect_final(self):
        """Inspect the final `HpBaseWorkChain`."""
        workchain = self.ctx.compute_hp

        if not workchain.is_finished_ok:
            self.report(f'final work chain {workchain} failed with status {workchain.exit_status}, aborting.')
            return self.exit_codes.ERROR_FINAL_WORKCHAIN_FAILED

    def results(self):
        """Retrieve the results from the final matrix collection workchain."""
        self.out_many(self.exposed_outputs(self.ctx.compute_hp, HpBaseWorkChain))

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
