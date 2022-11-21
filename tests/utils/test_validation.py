# -*- coding: utf-8 -*-
"""Tests for the :mod:`aiida_quantumespresso_hp.utils.validation` module."""
from aiida import orm
import pytest

from aiida_quantumespresso_hp.utils.validation import validate_parent_calculation


def test_validate_parent_calculation():
    """Test the `validate_parent_calculation` function for a valid input."""


@pytest.mark.usefixtures('aiida_profile_clean')
def test_validate_parent_calculation_raises(generate_calc_job_node, generate_structure):
    """Test the `validate_parent_calculation` function for a invalid input."""
    node = orm.Node()
    with pytest.raises(ValueError, match=r'parent calculation is not of type `PwCalculation` but .*'):
        validate_parent_calculation(node)

    node = generate_calc_job_node('quantumespresso.hp')
    with pytest.raises(ValueError, match=r'parent calculation is not of type `PwCalculation` but .*'):
        validate_parent_calculation(node)

    node = generate_calc_job_node('quantumespresso.pw')
    with pytest.raises(ValueError, match=r'could not retrieve the input parameters node'):
        validate_parent_calculation(node)

    inputs = {'parameters': orm.Dict(dict={})}
    node = generate_calc_job_node('quantumespresso.pw', inputs=inputs)
    with pytest.raises(ValueError, match=r'the parent calculation did not set `lda_plus_u=True`'):
        validate_parent_calculation(node)

    inputs = {'parameters': orm.Dict(dict={'SYSTEM': {'lda_plus_u': True}})}
    node = generate_calc_job_node('quantumespresso.pw', inputs=inputs)
    with pytest.raises(ValueError, match=r'the parent calculation did not specify any Hubbard U or V parameters'):
        validate_parent_calculation(node)

    inputs = {'parameters': orm.Dict(dict={'SYSTEM': {'lda_plus_u': True, 'hubbard_u': {'O': 1.0}}})}
    node = generate_calc_job_node('quantumespresso.pw', inputs=inputs)
    with pytest.raises(ValueError, match=r'could not retrieve the input structure node'):
        validate_parent_calculation(node)

    inputs = {
        'parameters': orm.Dict(dict={'SYSTEM': {
            'lda_plus_u': True,
            'hubbard_u': {
                'O': 1.0
            }
        }}),
        'structure': generate_structure()
    }
    node = generate_calc_job_node('quantumespresso.pw', inputs=inputs)
    with pytest.raises(ValueError, match=r'the structure does not have the right kind order'):
        validate_parent_calculation(node)

    inputs = {
        'parameters': orm.Dict(dict={'SYSTEM': {
            'lda_plus_u': True,
            'hubbard_u': {
                'Si': 1.0
            }
        }}),
        'structure': generate_structure()
    }
    node = generate_calc_job_node('quantumespresso.pw', inputs=inputs)
    validate_parent_calculation(node)
