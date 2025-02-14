# -*- coding: utf-8 -*-
"""Tests for the ``HpWorkChain.get_builder_from_protocol`` method."""
from aiida.engine import ProcessBuilder

from aiida_hubbard.workflows.hp.main import HpWorkChain


def test_get_available_protocols():
    """Test ``HpWorkChain.get_available_protocols``."""
    protocols = HpWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['fast', 'moderate', 'precise']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``HpWorkChain.get_default_protocol``."""
    assert HpWorkChain.get_default_protocol() == 'moderate'


def test_default(fixture_code, data_regression, serialize_builder):
    """Test ``HpWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code('quantumespresso.hp')

    builder = HpWorkChain.get_builder_from_protocol(code)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_parent_scf_folder(fixture_code, generate_calc_job_node, generate_inputs_pw, generate_hubbard_structure):
    """Test ``HpBaseWorkChain.get_builder_from_protocol`` with ``parent_scf_folder`` keyword."""
    code = fixture_code('quantumespresso.hp')
    inputs_pw = generate_inputs_pw()
    inputs_pw['structure'] = generate_hubbard_structure()
    parent_scf_folder = generate_calc_job_node('quantumespresso.pw', inputs=inputs_pw).outputs.remote_folder

    builder = HpWorkChain.get_builder_from_protocol(code, parent_scf_folder=parent_scf_folder)
    assert builder.hp.parent_scf == parent_scf_folder


def test_qpoints_overrides(fixture_code):
    """Test specifying qpoints ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.hp')
    overrides = {'qpoints': [1, 2, 3]}

    builder = HpWorkChain.get_builder_from_protocol(code, overrides=overrides)

    assert builder.qpoints.get_kpoints_mesh() == ([1, 2, 3], [0.0, 0.0, 0.0])


def test_options(fixture_code):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('quantumespresso.hp')

    queue_name = 'super-fast'
    withmpi = False  # The protocol default is ``True``

    options = {'queue_name': queue_name, 'withmpi': withmpi}
    builder = HpWorkChain.get_builder_from_protocol(code, options=options)

    assert builder.hp.metadata['options']['queue_name'] == queue_name
