# -*- coding: utf-8 -*-
"""Tests for the ``HpBaseWorkChain.get_builder_from_protocol`` method."""
from aiida.engine import ProcessBuilder

from aiida_hubbard.workflows.hp.base import HpBaseWorkChain


def test_get_available_protocols():
    """Test ``HpBaseWorkChain.get_available_protocols``."""
    protocols = HpBaseWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['fast', 'moderate', 'precise']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``HpBaseWorkChain.get_default_protocol``."""
    assert HpBaseWorkChain.get_default_protocol() == 'moderate'


def test_default(fixture_code, data_regression, serialize_builder):
    """Test ``HpBaseWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code('quantumespresso.hp')
    builder = HpBaseWorkChain.get_builder_from_protocol(code)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_parent_scf_folder(fixture_code, generate_calc_job_node, generate_inputs_pw, generate_hubbard_structure):
    """Test ``HpBaseWorkChain.get_builder_from_protocol`` with ``parent_scf_folder`` keyword."""
    code = fixture_code('quantumespresso.hp')
    inputs_pw = generate_inputs_pw()
    inputs_pw['structure'] = generate_hubbard_structure()
    parent_scf_folder = generate_calc_job_node('quantumespresso.pw', inputs=inputs_pw).outputs.remote_folder

    builder = HpBaseWorkChain.get_builder_from_protocol(code, parent_scf_folder=parent_scf_folder)
    assert builder.hp.parent_scf == parent_scf_folder


def test_parent_hp_folders(fixture_code, generate_calc_job_node):
    """Test ``HpBaseWorkChain.get_builder_from_protocol`` with ``parent_hp_folders`` keyword."""
    code = fixture_code('quantumespresso.hp')
    parent_hp_folders = {'site_01': generate_calc_job_node('quantumespresso.hp').outputs.retrieved}

    builder = HpBaseWorkChain.get_builder_from_protocol(code, parent_hp_folders=parent_hp_folders)
    assert 'parent_hp' in builder.hp
    assert 'site_01' in builder.hp.parent_hp
    assert builder.hp.parent_hp['site_01'] == parent_hp_folders['site_01']


def test_parameter_overrides(fixture_code):
    """Test specifying parameter ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.hp')

    overrides = {'hp': {'parameters': {'INPUTHP': {'conv_thr_chi': 1}}}}
    builder = HpBaseWorkChain.get_builder_from_protocol(code, overrides=overrides)
    assert builder.hp.parameters['INPUTHP']['conv_thr_chi'] == 1  # pylint: disable=no-member


def test_settings_overrides(fixture_code):
    """Test specifying settings ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.hp')

    overrides = {'hp': {'settings': {'cmdline': ['--kickass-mode']}}}
    builder = HpBaseWorkChain.get_builder_from_protocol(code, overrides=overrides)
    assert builder.hp.settings['cmdline'] == ['--kickass-mode']  # pylint: disable=no-member
    assert builder.hp.settings['parent_folder_symlink']


def test_metadata_overrides(fixture_code):
    """Test specifying metadata ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.hp')

    overrides = {'hp': {'metadata': {'options': {'resources': {'num_machines': 1e90}, 'max_wallclock_seconds': 1}}}}
    builder = HpBaseWorkChain.get_builder_from_protocol(
        code,
        overrides=overrides,
    )
    metadata = builder.hp.metadata  # pylint: disable=no-member

    assert metadata['options']['resources']['num_machines'] == 1e90
    assert metadata['options']['max_wallclock_seconds'] == 1


def test_options(fixture_code):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.hp')

    queue_name = 'super-fast'
    withmpi = False  # The protocol default is ``True``

    options = {'queue_name': queue_name, 'withmpi': withmpi}
    builder = HpBaseWorkChain.get_builder_from_protocol(code, options=options)
    metadata = builder.hp.metadata  # pylint: disable=no-member

    assert metadata['options']['queue_name'] == queue_name
    assert metadata['options']['withmpi'] == withmpi
