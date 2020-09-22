# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `SelfConsistentHubbardWorkChain` class."""
import pytest

from aiida.orm import Dict


@pytest.fixture
def generate_workchain_hubbard(generate_workchain, generate_inputs_hubbard):
    """Generate an instance of a `SelfConsistentHubbardWorkChain`."""

    def _generate_workchain_hubbard(inputs=None):
        entry_point = 'quantumespresso.hp.hubbard'

        if inputs is None:
            inputs = generate_inputs_hubbard()

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_hubbard


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_hubbard, generate_inputs_hubbard):
    """Test `SelfConsistentHubbardWorkChain.setup`."""
    inputs = generate_inputs_hubbard()
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()

    assert process.ctx.iteration == 0
    assert process.ctx.current_structure == inputs['structure']
    assert process.ctx.current_hubbard_u == inputs['hubbard_u'].get_dict()
    assert process.ctx.is_converged is False
    assert process.ctx.is_magnetic is None
    assert process.ctx.is_metal is None
    assert process.ctx.iteration == 0


@pytest.mark.usefixtures('aiida_profile')
def test_validate_inputs_invalid_structure(generate_workchain_hubbard, generate_inputs_hubbard, generate_structure):
    """Test `SelfConsistentHubbardWorkChain.validate_inputs`."""
    inputs = generate_inputs_hubbard()
    inputs['structure'] = generate_structure((('Li', 'Li'), ('Co', 'Co')))
    inputs['hubbard_u'] = Dict(dict={'Co': 1})

    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()
    process.validate_inputs()

    assert process.ctx.current_structure != inputs['structure']


@pytest.mark.usefixtures('aiida_profile')
def test_validate_inputs_valid_structure(generate_workchain_hubbard, generate_inputs_hubbard, generate_structure):
    """Test `SelfConsistentHubbardWorkChain.validate_inputs`."""
    inputs = generate_inputs_hubbard()
    inputs['structure'] = generate_structure((('Co', 'Co'), ('Li', 'Li')))
    inputs['hubbard_u'] = Dict(dict={'Co': 1})

    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()
    process.validate_inputs()

    assert process.ctx.current_structure == inputs['structure']
