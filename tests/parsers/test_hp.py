# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `HpParser`."""
from __future__ import absolute_import

import pytest
from aiida import orm
from aiida.common import AttributeDict


@pytest.fixture
def generate_inputs_default():
    """Return only those inputs that the parser will expect to be there."""
    parameters = {'INPUTHP': {}}
    qpoints = orm.KpointsData()
    qpoints.set_kpoints_mesh([1, 1, 1])

    return AttributeDict({
        'qpoints': qpoints,
        'parameters': orm.Dict(dict=parameters),
    })


def test_hp_default(fixture_database, fixture_computer_localhost, generate_calc_job_node, generate_parser,
                    generate_inputs_default, data_regression):
    """Test a default `hp.x` calculation."""
    name = 'default'
    entry_point_calc_job = 'quantumespresso.hp'
    entry_point_parser = 'quantumespresso.hp'

    node = generate_calc_job_node(entry_point_calc_job, fixture_computer_localhost, name, generate_inputs_default)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'chi' in results
    assert 'hubbard' in results
    assert 'matrices' in results
    assert 'parameters' in results
    data_regression.check({
        'chi': results['chi'].attributes,
        'hubbard': results['hubbard'].get_dict(),
        'matrices': results['matrices'].attributes,
        'parameters': results['parameters'].get_dict(),
    })
