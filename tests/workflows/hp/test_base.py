# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `HpBaseWorkChain` class."""
import pytest

from plumpy import ProcessState

from aiida.common import AttributeDict
from aiida.engine import ProcessHandlerReport

from aiida_quantumespresso_hp.calculations.hp import HpCalculation
from aiida_quantumespresso_hp.workflows.hp.base import HpBaseWorkChain


@pytest.fixture
def generate_workchain_hp(generate_workchain, generate_inputs_hp, generate_calc_job_node):
    """Generate an instance of a `HpBaseWorkChain`."""

    def _generate_workchain_hp(exit_code=None, inputs=None):
        entry_point = 'quantumespresso.hp.base'
        inputs = generate_inputs_hp(inputs=inputs)
        process = generate_workchain(entry_point, {'hp': inputs})

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
        ({}, {'alpha_mix(1)': 0.2, 'niter_max': 200}),
        ({'niter_max': 5}, {'alpha_mix(1)': 0.2, 'niter_max': 10}),
        ({'alpha_mix(5)': 0.5}, {'alpha_mix(5)': 0.25, 'niter_max': 200}),
        ({'alpha_mix(5)': 0.5, 'alpha_mix(10)': 0.4}, {'alpha_mix(5)': 0.25, 'alpha_mix(10)': 0.2, 'niter_max': 200}),
        ({'niter_max': 1, 'alpha_mix(2)': 0.3}, {'niter_max': 2, 'alpha_mix(2)': 0.15}),
    ),
)  # yapf: disable
def test_handle_convergence_not_reached(generate_workchain_hp, inputs, expected):
    """Test `HpBaseWorkChain.handle_convergence_not_reached`."""
    process = generate_workchain_hp(HpCalculation.exit_codes.ERROR_CONVERGENCE_NOT_REACHED, inputs)
    process.setup()
    process.validate_parameters()

    result = process.handle_convergence_not_reached(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break

    assert process.ctx.inputs.parameters['INPUTHP'] == expected
