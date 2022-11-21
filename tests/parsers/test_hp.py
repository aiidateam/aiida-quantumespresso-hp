# -*- coding: utf-8 -*-
# pylint: disable=unused-argument,redefined-outer-name
"""Tests for the `HpParser`."""

from aiida import orm
from aiida.common import AttributeDict
import pytest


@pytest.fixture
def generate_inputs_default():
    """Return only those inputs that the parser will expect to be there."""
    parameters = {'INPUTHP': {}}
    return AttributeDict({'parameters': orm.Dict(dict=parameters)})


@pytest.fixture
def generate_inputs_init_only():
    """Return only those inputs that the parser will expect to be there."""
    parameters = {'INPUTHP': {'determine_num_pert_only': True}}
    return AttributeDict({'parameters': orm.Dict(dict=parameters)})


@pytest.mark.usefixtures('aiida_profile_clean')
def test_hp_default(aiida_localhost, generate_calc_job_node, generate_parser, generate_inputs_default, data_regression):
    """Test a default `hp.x` calculation."""
    name = 'default'
    entry_point_calc_job = 'quantumespresso.hp'
    entry_point_parser = 'quantumespresso.hp'

    node = generate_calc_job_node(entry_point_calc_job, aiida_localhost, name, generate_inputs_default)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'parameters' in results
    assert 'hubbard' in results
    assert 'hubbard_chi' in results
    assert 'hubbard_matrices' in results
    data_regression.check({
        'parameters': results['parameters'].get_dict(),
        'hubbard': results['hubbard'].get_dict(),
        'hubbard_chi': results['hubbard_chi'].attributes,
        'hubbard_matrices': results['hubbard_matrices'].attributes,
    })


@pytest.mark.usefixtures('aiida_profile_clean')
def test_hp_initialization_only(
    aiida_localhost, generate_calc_job_node, generate_parser, generate_inputs_init_only, data_regression
):
    """Test an initialization only `hp.x` calculation."""
    name = 'initialization_only'
    entry_point_calc_job = 'quantumespresso.hp'
    entry_point_parser = 'quantumespresso.hp'

    node = generate_calc_job_node(entry_point_calc_job, aiida_localhost, name, generate_inputs_init_only)
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'parameters' in results
    assert 'hubbard' not in results
    assert 'hubbard_chi' not in results
    assert 'hubbard_matrices' not in results
    data_regression.check({
        'parameters': results['parameters'].get_dict(),
    })


@pytest.mark.usefixtures('aiida_profile_clean')
def test_hp_failed_invalid_namelist(aiida_localhost, generate_calc_job_node, generate_parser, generate_inputs_default):
    """Test an `hp.x` calculation that fails because of an invalid namelist."""
    name = 'failed_invalid_namelist'
    entry_point_calc_job = 'quantumespresso.hp'
    entry_point_parser = 'quantumespresso.hp'

    node = generate_calc_job_node(entry_point_calc_job, aiida_localhost, name, generate_inputs_default)
    parser = generate_parser(entry_point_parser)
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_INVALID_NAMELIST.status


@pytest.mark.usefixtures('aiida_profile_clean')
def test_failed_stdout_incomplete(generate_calc_job_node, generate_parser, generate_inputs_default, data_regression):
    """Test calculation that exited prematurely and so the stdout is incomplete."""
    name = 'failed_stdout_incomplete'
    entry_point_calc_job = 'quantumespresso.hp'
    entry_point_parser = 'quantumespresso.hp'

    node = generate_calc_job_node(entry_point_calc_job, test_name=name, inputs=generate_inputs_default)
    parser = generate_parser(entry_point_parser)
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE.status


@pytest.mark.usefixtures('aiida_profile_clean')
def test_failed_no_hubbard_parameters(
    generate_calc_job_node, generate_parser, generate_inputs_default, data_regression
):
    """Test calculation that did not generate the Hubbard parameters output file."""
    name = 'failed_no_hubbard_parameters'
    entry_point_calc_job = 'quantumespresso.hp'
    entry_point_parser = 'quantumespresso.hp'

    node = generate_calc_job_node(entry_point_calc_job, test_name=name, inputs=generate_inputs_default)
    parser = generate_parser(entry_point_parser)
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_OUTPUT_HUBBARD_MISSING.status


@pytest.mark.usefixtures('aiida_profile_clean')
def test_failed_no_hubbard_chi(generate_calc_job_node, generate_parser, generate_inputs_default, data_regression):
    """Test calculation that did not generate the Hubbard chi output file."""
    name = 'failed_no_hubbard_chi'
    entry_point_calc_job = 'quantumespresso.hp'
    entry_point_parser = 'quantumespresso.hp'

    node = generate_calc_job_node(entry_point_calc_job, test_name=name, inputs=generate_inputs_default)
    parser = generate_parser(entry_point_parser)
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status == node.process_class.exit_codes.ERROR_OUTPUT_HUBBARD_CHI_MISSING.status
