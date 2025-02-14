# -*- coding: utf-8 -*-
"""Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the Hubbard atoms."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, while_
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_hubbard.utils.general import is_perturb_only_atom

PwCalculation = CalculationFactory('quantumespresso.pw')
HpCalculation = CalculationFactory('quantumespresso.hp')
HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')


def validate_inputs(inputs, _):
    """Validate the top level namespace."""
    parameters = inputs['hp']['parameters'].get_dict().get('INPUTHP', {})

    if not bool(is_perturb_only_atom(parameters)):
        return 'The parameters in `hp.parameters` do not specify the required key `INPUTHP.pertub_only_atom`'


class HpParallelizeQpointsWorkChain(WorkChain):
    """Work chain to launch a Quantum Espresso hp.x calculation parallelizing over the q points on a single Hubbard atom."""  # pylint: disable=line-too-long

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpBaseWorkChain, exclude=('only_initialization', 'clean_workdir'))
        spec.input('max_concurrent_base_workchains', valid_type=orm.Int, required=False)
        spec.input(
            'init_walltime', valid_type=int, default=3600, non_db=True,
            help='The walltime of the initialization `HpBaseWorkChain` in seconds (default: 3600).'
            )
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.run_init,
            cls.inspect_init,
            while_(cls.should_run_qpoints)(
                cls.run_qpoints,
            ),
            cls.inspect_qpoints,
            cls.run_final,
            cls.results
        )
        spec.inputs.validator = validate_inputs
        spec.expose_outputs(HpBaseWorkChain)
        spec.exit_code(300, 'ERROR_QPOINT_WORKCHAIN_FAILED',
            message='A child work chain failed.')
        spec.exit_code(301, 'ERROR_INITIALIZATION_WORKCHAIN_FAILED',
            message='The child work chain failed.')
        spec.exit_code(302, 'ERROR_FINAL_WORKCHAIN_FAILED',
            message='The child work chain failed.')

    def run_init(self):
        """Run an initialization `HpBaseWorkChain` that will determine the number of perturbations (q points).

        This information is parsed and can be used to determine exactly how many
        `HpBaseWorkChains` have to be launched in parallel.
        """
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        parameters = inputs.hp.parameters.get_dict()
        parameters['INPUTHP']['determine_q_mesh_only'] = True
        inputs.hp.parameters = orm.Dict(parameters)
        inputs.clean_workdir = self.inputs.clean_workdir

        inputs.hp.metadata.options.max_wallclock_seconds =  self.inputs.init_walltime
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

        self.ctx.qpoints = list(range(workchain.outputs.parameters.dict.number_of_qpoints))

    def should_run_qpoints(self):
        """Return whether there are more q points to run."""
        return len(self.ctx.qpoints) > 0

    def run_qpoints(self):
        """Run a separate `HpBaseWorkChain` for each of the q points."""
        n_base_parallel = self.inputs.max_concurrent_base_workchains.value if 'max_concurrent_base_workchains' in self.inputs else len(self.ctx.qpoints)

        for _ in self.ctx.qpoints[:n_base_parallel]:
            qpoint_index = self.ctx.qpoints.pop(0)
            key = f'qpoint_{qpoint_index + 1}' # to keep consistency with QE
            inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
            inputs.clean_workdir = self.inputs.clean_workdir
            inputs.hp.parameters = inputs.hp.parameters.get_dict()
            inputs.hp.parameters['INPUTHP']['start_q'] = qpoint_index + 1 # QuantumESPRESSO starts from 1
            inputs.hp.parameters['INPUTHP']['last_q'] = qpoint_index + 1
            inputs.hp.parameters = orm.Dict(dict=inputs.hp.parameters)
            inputs.metadata.call_link_label = key

            node = self.submit(HpBaseWorkChain, **inputs)
            self.to_context(**{key: node})
            name = HpBaseWorkChain.__name__
            self.report(f'launched {name}<{node.pk}> for q point {qpoint_index}')

    def inspect_qpoints(self):
        """Inspect each parallel qpoint `HpBaseWorkChain`."""
        for key, workchain in self.ctx.items():
            if key.startswith('qpoint_'):
                if not workchain.is_finished_ok:
                    self.report(f'child work chain {workchain} failed with status {workchain.exit_status}, aborting.')
                    return self.exit_codes.ERROR_QPOINT_WORKCHAIN_FAILED

    def run_final(self):
        """Perform the final HpCalculation to collect the various components of the chi matrices."""
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.hp.parent_scf = inputs.hp.parent_scf
        inputs.hp.parent_hp = {key: wc.outputs.retrieved for key, wc in self.ctx.items() if key.startswith('qpoint_')}
        inputs.hp.metadata.options.max_wallclock_seconds = 3600 # 1 hour is more than enough
        inputs.metadata.call_link_label = 'compute_chi'

        node = self.submit(HpBaseWorkChain, **inputs)
        self.to_context(compute_chi=node)
        self.report(f'launched HpBaseWorkChain<{node.pk}> to collect perturbation matrices')

    def inspect_final(self):
        """Inspect the final `HpBaseWorkChain`."""
        workchain = self.ctx.compute_chi

        if not workchain.is_finished_ok:
            self.report(f'final work chain {workchain} failed with status {workchain.exit_status}, aborting.')
            return self.exit_codes.ERROR_FINAL_WORKCHAIN_FAILED

    def results(self):
        """Retrieve the results from the final matrix collection workchain."""
        self.out_many(self.exposed_outputs(self.ctx.compute_chi, HpBaseWorkChain))

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
