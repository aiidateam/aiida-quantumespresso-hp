# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `HpBaseWorkChain` class."""
from aiida.common import AttributeDict
from aiida.engine import ProcessHandlerReport
from plumpy import ProcessState
import pytest

from aiida_quantumespresso_hp.calculations.hp import HpCalculation
from aiida_quantumespresso_hp.workflows.hp.base import HpBaseWorkChain


@pytest.fixture
def generate_workchain_hp(generate_workchain, generate_inputs_hp, generate_calc_job_node):
    """Generate an instance of a `HpBaseWorkChain`."""

    def _generate_workchain_hp(exit_code=None, inputs=None, return_inputs=False):
        entry_point = 'quantumespresso.hp.base'

        if inputs is None:
            inputs = {'hp': generate_inputs_hp()}

        if return_inputs:
            return inputs

        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            node = generate_calc_job_node('quantumespresso.hp')
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_hp


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_hp):
    """Test `HpBaseWorkChain.setup`."""
    process = generate_workchain_hp()
    process.setup()

    assert process.ctx.restart_calc is None
    assert isinstance(process.ctx.inputs, AttributeDict)


def test_set_max_seconds(generate_workchain_hp):
    """Test that `max_seconds` gets set in the parameters based on `max_wallclock_seconds` unless already set."""
    inputs = generate_workchain_hp(return_inputs=True)
    max_wallclock_seconds = inputs['hp']['metadata']['options']['max_wallclock_seconds']

    process = generate_workchain_hp(inputs=inputs)
    process.setup()
    process.validate_parameters()
    process.prepare_process()

    expected_max_seconds = max_wallclock_seconds * process.defaults.delta_factor_max_seconds
    assert 'max_seconds' in process.ctx.inputs['parameters']['INPUTHP']
    assert process.ctx.inputs['parameters']['INPUTHP']['max_seconds'] == expected_max_seconds

    # Now check that if `max_seconds` is already explicitly set in the parameters, it is not overwritten.
    inputs = generate_workchain_hp(return_inputs=True)
    max_seconds = 1
    max_wallclock_seconds = inputs['hp']['metadata']['options']['max_wallclock_seconds']
    inputs['hp']['parameters']['INPUTHP']['max_seconds'] = max_seconds

    process = generate_workchain_hp(inputs=inputs)
    process.setup()
    process.validate_parameters()
    process.prepare_process()

    assert 'max_seconds' in process.ctx.inputs['parameters']['INPUTHP']
    assert process.ctx.inputs['parameters']['INPUTHP']['max_seconds'] == max_seconds


@pytest.mark.usefixtures('aiida_profile')
def test_handle_unrecoverable_failure(generate_workchain_hp):
    """Test `HpBaseWorkChain.handle_unrecoverable_failure`."""
    process = generate_workchain_hp(exit_code=HpCalculation.exit_codes.ERROR_NO_RETRIEVED_FOLDER)
    process.setup()

    result = process.handle_unrecoverable_failure(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code == HpBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE

    result = process.inspect_process()
    assert result == HpBaseWorkChain.exit_codes.ERROR_UNRECOVERABLE_FAILURE


@pytest.mark.usefixtures('aiida_profile')
@pytest.mark.parametrize(
    ('inputs', 'expected'),
    (
        ({}, {'alpha_mix(1)': 0.2}),
        ({'alpha_mix(5)': 0.5}, {'alpha_mix(5)': 0.25}),
        ({'alpha_mix(5)': 0.5, 'alpha_mix(10)': 0.4}, {'alpha_mix(5)': 0.25, 'alpha_mix(10)': 0.2}),
    ),
)  # yapf: disable
def test_handle_convergence_not_reached(generate_workchain_hp, generate_inputs_hp, inputs, expected):
    """Test `HpBaseWorkChain.handle_convergence_not_reached`."""
    inputs_hp = {'hp': generate_inputs_hp(inputs=inputs)}
    process = generate_workchain_hp(exit_code=HpCalculation.exit_codes.ERROR_CONVERGENCE_NOT_REACHED, inputs=inputs_hp)
    process.setup()
    process.validate_parameters()

    result = process.handle_convergence_not_reached(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break

    assert process.ctx.inputs.parameters['INPUTHP'] == expected


# yapf: disable
@pytest.mark.usefixtures('aiida_profile')
@pytest.mark.parametrize(
    ('cmdline', 'expected'),
    (
        ([], ['-nd', '1']),
        (['-nd', '2'], ['-nd', '1']),
        (['-nk', '2', '-nd', '2'], ['-nk', '2', '-nd', '1']),
        (['-nk', '2'], ['-nk', '2', '-nd', '1']),
    ),
)
# yapf: enable
def test_handle_computing_cholesky(generate_workchain_hp, generate_inputs_hp, cmdline, expected):
    """Test `HpBaseWorkChain.handle_computing_cholesky`."""
    from aiida.orm import Dict

    inputs_hp = {'hp': generate_inputs_hp()}
    inputs_hp['hp']['settings'] = Dict({'cmdline': cmdline})

    process = generate_workchain_hp(exit_code=HpCalculation.exit_codes.ERROR_COMPUTING_CHOLESKY, inputs=inputs_hp)
    process.setup()
    process.validate_parameters()

    result = process.handle_computing_cholesky(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break

    assert process.ctx.inputs.settings['cmdline'] == expected


def test_handle_computing_cholesky_fail(generate_workchain_hp, generate_inputs_hp):
    """Test `HpBaseWorkChain.handle_computing_cholesky` failing."""
    from aiida.orm import Dict

    inputs_hp = {'hp': generate_inputs_hp()}
    inputs_hp['hp']['settings'] = Dict({'cmdline': ['-nd', '1']})

    process = generate_workchain_hp(exit_code=HpCalculation.exit_codes.ERROR_COMPUTING_CHOLESKY, inputs=inputs_hp)
    process.setup()
    process.validate_parameters()

    result = process.handle_computing_cholesky(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert not result.do_break
