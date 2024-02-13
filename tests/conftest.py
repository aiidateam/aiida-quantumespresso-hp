# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name, too-many-statements
"""Initialise a text database and profile for pytest."""
from collections.abc import Mapping
import io
import os
import pathlib
import shutil
import tempfile

import pytest

pytest_plugins = ['aiida.manage.tests.pytest_fixtures']  # pylint: disable=invalid-name


@pytest.fixture(scope='session')
def filepath_tests():
    """Return the absolute filepath of the `tests` folder.

    .. warning:: if this file moves with respect to the `tests` folder, the implementation should change.

    :return: absolute filepath of `tests` folder which is the basepath for all test resources.
    """
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return os.path.join(filepath_tests, 'fixtures')


@pytest.fixture(scope='function')
def fixture_sandbox_folder():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return an ``InstalledCode`` instance configured to run calculations of given entry point on localhost."""

    def _fixture_code(entry_point_name):
        from aiida.common import exceptions
        from aiida.orm import InstalledCode, load_code

        label = f'test.{entry_point_name}'

        try:
            return load_code(label=label)
        except (exceptions.NotExistent, exceptions.MultipleObjectsError):
            return InstalledCode(
                label=label,
                computer=fixture_localhost,
                filepath_executable='/bin/true',
                default_calc_job_plugin=entry_point_name,
            )

    return _fixture_code


@pytest.fixture
def serialize_builder():
    """Serialize the given process builder into a dictionary with nodes turned into their value representation.

    :param builder: the process builder to serialize
    :return: dictionary
    """

    def serialize_data(data):
        # pylint: disable=too-many-return-statements
        from aiida.orm import AbstractCode, BaseType, Data, Dict, KpointsData, List, RemoteData, SinglefileData
        from aiida.plugins import DataFactory

        StructureData = DataFactory('core.structure')
        UpfData = DataFactory('pseudo.upf')

        if isinstance(data, dict):
            return {key: serialize_data(value) for key, value in data.items()}

        if isinstance(data, BaseType):
            return data.value

        if isinstance(data, AbstractCode):
            return data.full_label

        if isinstance(data, Dict):
            return data.get_dict()

        if isinstance(data, List):
            return data.get_list()

        if isinstance(data, StructureData):
            return data.get_formula()

        if isinstance(data, UpfData):
            return f'{data.element}<md5={data.md5}>'

        if isinstance(data, RemoteData):
            # For `RemoteData` we compute the hash of the repository. The value returned by `Node._get_hash` is not
            # useful since it includes the hash of the absolute filepath and the computer UUID which vary between tests
            return data.base.repository.hash()

        if isinstance(data, KpointsData):
            try:
                return data.get_kpoints()
            except AttributeError:
                return data.get_kpoints_mesh()

        if isinstance(data, SinglefileData):
            return data.get_content()

        if isinstance(data, Data):
            return data.base.caching._get_hash()  # pylint: disable=protected-access

        return data

    def _serialize_builder(builder):
        return serialize_data(builder._inputs(prune=True))  # pylint: disable=protected-access

    return _serialize_builder


@pytest.fixture(scope='session', autouse=True)
def sssp(aiida_profile, generate_upf_data):
    """Create an SSSP pseudo potential family from scratch."""
    from aiida.common.constants import elements
    from aiida.plugins import GroupFactory

    aiida_profile.clear_profile()

    SsspFamily = GroupFactory('pseudo.family.sssp')

    cutoffs = {}
    stringency = 'standard'

    with tempfile.TemporaryDirectory() as dirpath:
        for values in elements.values():

            element = values['symbol']

            actinides = ('Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr')

            if element in actinides:
                continue

            upf = generate_upf_data(element)
            dirpath = pathlib.Path(dirpath)
            filename = dirpath / f'{element}.upf'

            with open(filename, 'w+b') as handle:
                with upf.open(mode='rb') as source:
                    handle.write(source.read())
                    handle.flush()

            cutoffs[element] = {
                'cutoff_wfc': 30.0,
                'cutoff_rho': 240.0,
            }

        label = 'SSSP/1.3/PBEsol/efficiency'
        family = SsspFamily.create_from_folder(dirpath, label)

    family.set_cutoffs(cutoffs, stringency, unit='Ry')

    return family


@pytest.fixture
def generate_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was
    passed to it, into which the raw input files will have been written.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job


@pytest.fixture
def generate_calc_job_node(aiida_localhost):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""

    def flatten_inputs(inputs, prefix=''):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + '__'))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(
        entry_point_name, computer=None, test_name=None, inputs=None, attributes=None, retrieve_temporary=None
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :param attributes: any optional attributes to set on the node
        :param retrieve_temporary: optional tuple of an absolute filepath of a temporary directory and a list of
            filenames that should be written to this directory, which will serve as the `retrieved_temporary_folder`.
            For now this only works with top-level files and does not support files nested in directories.
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node
        """
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = aiida_localhost

        filepath_folder = None

        if test_name is not None:
            basepath = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(entry_point_name[len('quantumespresso.'):], test_name)
            filepath_folder = os.path.join(basepath, 'parsers', 'fixtures', filename)

        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.base.attributes.set('input_filename', 'aiida.in')
        node.base.attributes.set('output_filename', 'aiida.out')
        node.base.attributes.set('error_filename', 'aiida.err')
        node.set_option('resources', {'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        node.set_option('max_wallclock_seconds', 1800)

        if attributes:
            node.set_attribute_many(attributes)

        if inputs:
            metadata = inputs.pop('metadata', {})
            options = metadata.get('options', {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.base.links.add_incoming(input_node, link_type=LinkType.INPUT_CALC, link_label=link_label)

        node.store()

        if retrieve_temporary:
            dirpath, filenames = retrieve_temporary
            for filename in filenames:
                try:
                    shutil.copy(os.path.join(filepath_folder, filename), os.path.join(dirpath, filename))
                except FileNotFoundError:
                    pass  # To test the absence of files in the retrieve_temporary folder

        retrieved = orm.FolderData()

        if filepath_folder:
            retrieved.base.repository.put_object_from_tree(filepath_folder)

            # Remove files that are supposed to be only present in the retrieved temporary folder
            if retrieve_temporary:
                for filename in filenames:
                    try:
                        retrieved.base.repository.delete_object(filename)
                    except OSError:
                        pass  # To test the absence of files in the retrieve_temporary folder

        retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label='retrieved')
        retrieved.store()

        remote_folder = orm.RemoteData(computer=computer, remote_path='/tmp')
        remote_folder.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label='remote_folder')
        remote_folder.store()

        return node

    return _generate_calc_job_node


@pytest.fixture
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain


@pytest.fixture
def generate_structure():
    """Return a `StructureData` representing bulk silicon."""

    def _generate_structure(structure_id=None):
        """Return a `StructureData` representing bulk silicon."""
        from aiida.orm import StructureData

        if structure_id is None:
            sites = [('Si', 'Si')]

            cell = [[1., 1., 0], [1., 0, 1.], [0, 1., 1.]]
            structure = StructureData(cell=cell)

            for kind, symbol in sites:
                structure.append_atom(position=(0., 0., 0.), symbols=symbol, name=kind)

        if structure_id == 'licoo2':
            # LiCoO2 structure used in several QuantumESPRESSO HP examples.
            a, b, c, d = 1.40803, 0.81293, 4.68453, 1.62585
            cell = [[a, -b, c], [0.0, d, c], [-a, -b, c]]
            positions = [[0, 0, 0], [0, 0, 3.6608], [0, 0, 10.392], [0, 0, 7.0268]]
            symbols = ['Co', 'O', 'O', 'Li']
            structure = StructureData(cell=cell)
            for position, symbol in zip(positions, symbols):
                structure.append_atom(position=position, symbols=symbol)

        if structure_id == 'AFMlicoo2':
            # LiCoO2 with 4 Co atoms
            # Unrealistic structure - just for testing AFM sublattices
            a, b, c, d = 1.40803, 0.81293, 4.68453, 1.62585
            cell = [[a, -b, c], [0.0, d, c], [-a, -b, c]]
            positions = [[0, 0, 0], [0, 0, 1.5], [0, 0, -1.5], [0, 0, 0.5], [0, 0, 3.6608], [0, 0, 10.392],
                         [0, 0, 7.0268]]
            names = ['Co0', 'Co0', 'Co1', 'Co1', 'O', 'O', 'Li']
            symbols = ['Co', 'Co', 'Co', 'Co', 'O', 'O', 'Li']
            structure = StructureData(cell=cell)
            for position, symbol, name in zip(positions, symbols, names):
                structure.append_atom(position=position, symbols=symbol, name=name)

        return structure

    return _generate_structure


@pytest.fixture
def generate_hubbard_structure(generate_structure):
    """Return a `HubbardStructureData` representing bulk silicon."""

    def _generate_hubbard_structure(only_u=False, u_value=1e-5, v_value=1e-5):
        """Return a `StructureData` representing bulk silicon."""
        from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

        structure = generate_structure(structure_id='licoo2')
        hubbard_structure = HubbardStructureData.from_structure(structure=structure)

        if only_u:
            hubbard_structure.initialize_onsites_hubbard('Co', '3d', u_value)
        else:
            hubbard_structure.initialize_onsites_hubbard('Co', '3d', u_value)
            hubbard_structure.initialize_intersites_hubbard('Co', '3d', 'O', '2p', v_value)

        return hubbard_structure

    return _generate_hubbard_structure


@pytest.fixture
def generate_kpoints_mesh():
    """Return a `KpointsData` node."""

    def _generate_kpoints_mesh(npoints):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3)

        return kpoints

    return _generate_kpoints_mesh


@pytest.fixture
def generate_parser():
    """Fixture to load a parser class for testing parsers."""

    def _generate_parser(entry_point_name):
        """Fixture to load a parser class for testing parsers.

        :param entry_point_name: entry point name of the parser class
        :return: the `Parser` sub class
        """
        from aiida.plugins import ParserFactory
        return ParserFactory(entry_point_name)

    return _generate_parser


@pytest.fixture
def generate_inputs_pw(fixture_code, generate_structure, generate_kpoints_mesh, generate_upf_data):
    """Generate default inputs for a `PwCalculation."""

    def _generate_inputs_pw(parameters=None, structure=None):
        """Generate default inputs for a `PwCalculation."""
        from aiida.orm import Dict
        from aiida_quantumespresso.utils.resources import get_default_options

        parameters_base = {'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutrho': 240.0, 'ecutwfc': 30.0}}

        if parameters is not None:
            parameters_base.update(parameters)
        structure = structure or generate_structure()
        inputs = {
            'code': fixture_code('quantumespresso.pw'),
            'structure': structure,
            'kpoints': generate_kpoints_mesh(2),
            'parameters': Dict(parameters_base),
            'pseudos': {kind: generate_upf_data(kind) for kind in structure.get_kind_names()},
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_pw


@pytest.fixture
def generate_inputs_hp(
    fixture_code, aiida_localhost, generate_calc_job_node, generate_inputs_pw, generate_kpoints_mesh,
    generate_hubbard_structure
):
    """Generate default inputs for a `HpCalculation."""

    def _generate_inputs_hp(inputs=None):
        """Generate default inputs for a `HpCalculation."""
        from aiida.orm import Dict
        from aiida_quantumespresso.utils.resources import get_default_options

        hubbard_structure = generate_hubbard_structure()

        parent_inputs = generate_inputs_pw(structure=hubbard_structure)
        parent = generate_calc_job_node('quantumespresso.pw', aiida_localhost, inputs=parent_inputs)
        inputs = {
            'code': fixture_code('quantumespresso.hp'),
            'parent_scf': parent.outputs.remote_folder,
            'qpoints': generate_kpoints_mesh(2),
            'parameters': Dict({'INPUTHP': inputs or {}}),
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_hp


@pytest.fixture
def generate_inputs_hubbard(generate_inputs_pw, generate_inputs_hp, generate_hubbard_structure):
    """Generate default inputs for a `SelfConsistentHubbardWorkChain."""

    def _generate_inputs_hubbard(hubbard_structure=None):
        """Generate default inputs for a `SelfConsistentHubbardWorkChain."""
        from aiida.orm import Bool

        hubbard_structure = hubbard_structure or generate_hubbard_structure()
        inputs_pw = generate_inputs_pw(structure=hubbard_structure)
        inputs_relax = generate_inputs_pw(structure=hubbard_structure)
        inputs_hp = generate_inputs_hp()

        kpoints = inputs_pw.pop('kpoints')
        inputs_pw.pop('structure')

        inputs_relax.pop('kpoints')
        inputs_relax.pop('structure')

        inputs_hp.pop('parent_scf')

        inputs = {
            'meta_convergence': Bool(True),
            'hubbard_structure': hubbard_structure,
            'relax': {
                'base': {
                    'pw': inputs_pw,
                    'kpoints': kpoints,
                }
            },
            'scf': {
                'pw': inputs_pw,
                'kpoints': kpoints,
            },
            'hubbard': {
                'hp': inputs_hp,
            }
        }

        return inputs

    return _generate_inputs_hubbard


@pytest.fixture
def generate_hp_retrieved():
    """Generate a `FolderData` that acts as the `retrieved` output of a partial `HpCalculation."""
    from aiida.common import LinkType
    from aiida.orm import CalcJobNode, FolderData
    from aiida.plugins import CalculationFactory
    from aiida.plugins.entry_point import format_entry_point_string

    HpCalculation = CalculationFactory('quantumespresso.hp')

    process_type = format_entry_point_string('aiida.calculations', 'quantumespresso.hp')
    filename = os.path.join(HpCalculation.dirname_output_hubbard, 'aiida.chi.pert_1.dat')

    calcjob = CalcJobNode(process_type=process_type).store()
    retrieved = FolderData()
    retrieved.base.repository.put_object_from_filelike(io.StringIO('pert'), filename)
    retrieved.base.links.add_incoming(calcjob, link_type=LinkType.CREATE, link_label='retrieved')
    retrieved.store()

    return retrieved


@pytest.fixture(scope='session')
def generate_upf_data():
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element):
        """Return `UpfData` node."""
        from aiida_pseudo.data.pseudo import UpfData
        content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
        stream = io.BytesIO(content.encode('utf-8'))
        return UpfData(stream, filename=f'{element}.upf')

    return _generate_upf_data
