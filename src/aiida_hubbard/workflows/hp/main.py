# -*- coding: utf-8 -*-
"""Work chain to run a Quantum ESPRESSO hp.x calculation."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, if_
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')
HpParallelizeAtomsWorkChain = WorkflowFactory('quantumespresso.hp.parallelize_atoms')


def validate_inputs(inputs, _):
    """Validate the top level namespace."""
    if inputs['parallelize_qpoints']:
        if not inputs['parallelize_atoms']:
            return 'To use `parallelize_qpoints`, also `parallelize_atoms` must be `True`'


class HpWorkChain(WorkChain, ProtocolMixin):
    """Work chain to run a Quantum ESPRESSO hp.x calculation.

    If the `parallelize_atoms` input is set to `True`, the calculation will be parallelized over the Hubbard atoms by
    running the `HpParallelizeAtomsWorkChain`. When parallelizing over atoms, if the `parallelize_qpoints` is `True`,
    each `HpParallelizeAtomsWorkChain` will be parallelized over its perturbations (q points) running the
    `HpParallelizeQpointsWorkChain`. Otherwise a single `HpBaseWorkChain` will be launched that will compute
    every Hubbard atom, and every q point in serial.

    .. important:: q point parallelization is only possible when parallelization over atoms is performed.
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(HpBaseWorkChain, exclude=('clean_workdir', 'only_initialization', 'hp.qpoints'))
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.input('qpoints', valid_type=orm.KpointsData, required=False,
            help='An explicit q-points list or mesh. Either this or `qpoints_distance` has to be provided.')
        spec.input('qpoints_distance', valid_type=orm.Float, required=False,
            help='The minimum desired distance in 1/Å between q-points in reciprocal space. The explicit q-points will '
            'be generated automatically by a calculation function based on the input structure.')
        spec.input('qpoints_force_parity', valid_type=orm.Bool, required=False,
            help='Optional input when constructing the q-points based on a desired `qpoints_distance`. Setting this to '
            '`True` will force the q-point mesh to have an even number of points along each lattice vector except '
            'for any non-periodic directions.')
        spec.input('parallelize_atoms', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('parallelize_qpoints', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.input('max_concurrent_base_workchains', valid_type=orm.Int, required=False)
        spec.outline(
            cls.validate_qpoints,
            if_(cls.should_parallelize_atoms)(
                cls.run_parallel_workchain,
            ).else_(
                cls.run_base_workchain,
            ),
            cls.inspect_workchain,
            cls.results,
        )
        spec.inputs.validator = validate_inputs
        spec.expose_outputs(HpBaseWorkChain)
        spec.exit_code(200, 'ERROR_INVALID_INPUT_QPOINTS',
            message=('Neither the `qpoints` nor the `qpoints_distance`, '
                'or the `hp.hubbard_structure` input were specified.'))
        spec.exit_code(300, 'ERROR_CHILD_WORKCHAIN_FAILED', message='A child work chain failed.')

    @classmethod
    def get_protocol_filepath(cls):
        """Return ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from ..protocols import hp as hp_protocols
        return files(hp_protocols) / 'main.yaml'

    @classmethod
    def get_builder_from_protocol(cls, code, protocol=None, parent_scf_folder=None, overrides=None, options=None, **_):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.hp`` plugin.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param parent_scf_folder: the parent ``RemoteData`` of the respective SCF calcualtion.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        inputs = cls.get_protocol_inputs(protocol, overrides)

        data = HpBaseWorkChain.get_builder_from_protocol(  # pylint: disable=protected-access
            code, protocol=protocol, parent_scf_folder=parent_scf_folder, overrides=inputs, options=options, **_
        )._data

        data.pop('only_initialization', None)
        data['hp'].pop('qpoints', None)

        if 'qpoints' in inputs:
            qpoints = orm.KpointsData()
            qpoints.set_kpoints_mesh(inputs['qpoints'])
            data['qpoints'] = qpoints
        if 'qpoints_distance' in inputs:
            data['qpoints_distance'] = orm.Float(inputs['qpoints_distance'])
        if 'qpoints_force_parity' in inputs:
            data['qpoints_force_parity'] = orm.Bool(inputs['qpoints_force_parity'])
        if 'parallelize_atoms' in inputs:
            data['parallelize_atoms'] = orm.Bool(inputs['parallelize_atoms'])
        if 'parallelize_qpoints' in inputs:
            data['parallelize_qpoints'] = orm.Bool(inputs['parallelize_qpoints'])
        if 'max_concurrent_base_workchains' in inputs:
            data['max_concurrent_base_workchains'] = orm.Int(inputs['max_concurrent_base_workchains'])

        builder = cls.get_builder()
        builder._data = data  # pylint: disable=protected-access
        builder.clean_workdir = orm.Bool(inputs['clean_workdir'])

        return builder

    def validate_qpoints(self):
        """Validate the inputs related to q-points.

        Either an explicit `KpointsData` with given mesh/path, or a desired q-points distance should be specified. In
        the case of the latter, the `KpointsData` will be constructed for the input `StructureData` using the
        `create_kpoints_from_distance` calculation function.
        """
        from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import (
            create_kpoints_from_distance,
        )

        if all(key not in self.inputs for key in ['qpoints', 'qpoints_distance']):
            return self.exit_codes.ERROR_INVALID_INPUT_QPOINTS

        try:
            qpoints = self.inputs.qpoints
        except AttributeError:
            if 'hubbard_structure' in self.inputs.hp:
                inputs = {
                    'structure': self.inputs.hp.hubbard_structure,
                    'distance': self.inputs.qpoints_distance,
                    'force_parity': self.inputs.get('qpoints_force_parity', orm.Bool(False)),
                    'metadata': {
                        'call_link_label': 'create_qpoints_from_distance'
                    }
                }
                qpoints = create_kpoints_from_distance(**inputs)  # pylint: disable=unexpected-keyword-arg
            else:
                return self.exit_codes.ERROR_INVALID_INPUT_QPOINTS

        self.ctx.qpoints = qpoints

    def should_parallelize_atoms(self):
        """Return whether the calculation should be parallelized over atoms."""
        return self.inputs.parallelize_atoms.value

    def run_base_workchain(self):
        """Run the `HpBaseWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.hp.qpoints = self.ctx.qpoints
        running = self.submit(HpBaseWorkChain, **inputs)
        self.report(f'running in serial, launching HpBaseWorkChain<{running.pk}>')
        return ToContext(workchain=running)

    def run_parallel_workchain(self):
        """Run the `HpParallelizeAtomsWorkChain`."""
        inputs = AttributeDict(self.exposed_inputs(HpBaseWorkChain))
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.parallelize_qpoints = self.inputs.parallelize_qpoints
        inputs.hp.qpoints = self.ctx.qpoints
        if 'max_concurrent_base_workchains' in self.inputs:
            inputs.max_concurrent_base_workchains = self.inputs.max_concurrent_base_workchains
        running = self.submit(HpParallelizeAtomsWorkChain, **inputs)
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
