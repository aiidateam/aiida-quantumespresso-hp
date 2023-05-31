# -*- coding: utf-8 -*-
"""Tests for the ``SelfConsistentHubbardWorkChain.get_builder_from_protocol`` method."""
from aiida.engine import ProcessBuilder
import pytest

from aiida_quantumespresso_hp.workflows.hubbard import SelfConsistentHubbardWorkChain


def test_get_available_protocols():
    """Test ``SelfConsistentHubbardWorkChain.get_available_protocols``."""
    protocols = SelfConsistentHubbardWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['fast', 'moderate', 'precise']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``SelfConsistentHubbardWorkChain.get_default_protocol``."""
    assert SelfConsistentHubbardWorkChain.get_default_protocol() == 'moderate'


def test_default(fixture_code, data_regression, generate_hubbard_structure, serialize_builder):
    """Test ``SelfConsistentHubbardWorkChain.get_builder_from_protocol`` for the default protocol."""
    pw_code = fixture_code('quantumespresso.pw')
    hp_code = fixture_code('quantumespresso.hp')
    hubbard_structure = generate_hubbard_structure()

    builder = SelfConsistentHubbardWorkChain.get_builder_from_protocol(pw_code, hp_code, hubbard_structure)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    'overrides', (
        {
            'tolerance_onsite': 1
        },
        {
            'tolerance_intersite': 1
        },
        {
            'skip_relax_iterations': 2
        },
        {
            'relax_frequency': 3
        },
        {
            'max_iterations': 1
        },
        {
            'meta_convergence': False
        },
        {
            'clean_workdir': False
        },
    )
)
def test_overrides(fixture_code, generate_hubbard_structure, overrides):
    """Test specifying different``overrides`` for the ``get_builder_from_protocol()`` method."""
    pw_code = fixture_code('quantumespresso.pw')
    hp_code = fixture_code('quantumespresso.hp')
    hubbard_structure = generate_hubbard_structure()

    builder = SelfConsistentHubbardWorkChain.get_builder_from_protocol(
        pw_code, hp_code, hubbard_structure, overrides=overrides
    )

    for key, value in overrides.items():
        assert builder[key].value == value


def test_options(fixture_code, generate_hubbard_structure):
    """Test specifying ``options`` for the ``get_builder_from_protocol()`` method."""
    pw_code = fixture_code('quantumespresso.pw')
    hp_code = fixture_code('quantumespresso.hp')
    hubbard_structure = generate_hubbard_structure()

    queue_name = 'super-fast'
    withmpi = False  # The protocol default is ``True``

    options = {'queue_name': queue_name, 'withmpi': withmpi}
    builder = SelfConsistentHubbardWorkChain.get_builder_from_protocol(
        pw_code, hp_code, hubbard_structure, options_pw=options, options_hp=options
    )

    assert builder.hubbard.hp.metadata['options']['queue_name'] == queue_name
    assert builder.scf.pw.metadata['options']['queue_name'] == queue_name
    assert builder.relax.base.pw.metadata['options']['queue_name'] == queue_name
