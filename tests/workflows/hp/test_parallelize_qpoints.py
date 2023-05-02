# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `HpParallelizeQpointsWorkChain` class."""
from plumpy import ProcessState
import pytest

from aiida_quantumespresso_hp.workflows.hp.parallelize_qpoints import HpParallelizeQpointsWorkChain


@pytest.fixture
def generate_workchain_qpoints(generate_workchain, generate_inputs_hp, generate_hubbard_structure):
    """Generate an instance of a `HpParallelizeQpointsWorkChain`."""

    def _generate_workchain_qpoints(inputs=None):
        entry_point = 'quantumespresso.hp.parallelize_qpoints'

        if inputs is None:
            inputs = {'perturb_only_atom(1)': True}

        inputs = generate_inputs_hp(inputs=inputs)
        inputs['hubbard_structure'] = generate_hubbard_structure()
        process = generate_workchain(entry_point, {'hp': inputs})

        return process

    return _generate_workchain_qpoints


@pytest.fixture
def generate_hp_workchain_node(generate_calc_job_node):
    """Generate an instance of `WorkflowNode`."""

    def _generate_hp_workchain_node(exit_status=0, use_retrieved=False):
        from aiida.common import LinkType
        from aiida.orm import Dict, WorkflowNode

        node = WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        parameters = Dict({'number_of_qpoints': 2}).store()
        parameters.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='parameters')

        if use_retrieved:
            retrieved = generate_calc_job_node(
                'quantumespresso.hp'
            ).outputs.retrieved  # otherwise the HpCalculation will complain
            retrieved.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='retrieved')

        return node

    return _generate_hp_workchain_node


def test_validate_inputs_invalid_parameters(generate_workchain_qpoints):
    """Test `HpParallelizeQpointsWorkChain.validate_inputs`."""
    match = r'The parameters in `hp.parameters` do not specify the required key `INPUTHP.pertub_only_atom`'
    with pytest.raises(ValueError, match=match):
        generate_workchain_qpoints(inputs={})


@pytest.mark.usefixtures('aiida_profile')
def test_run_init(generate_workchain_qpoints):
    """Test `HpParallelizeQpointsWorkChain.run_init`."""
    process = generate_workchain_qpoints()
    process.run_init()

    assert 'initialization' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_qpoints(generate_workchain_qpoints, generate_hp_workchain_node):
    """Test `HpParallelizeQpointsWorkChain.run_qpoints`."""
    process = generate_workchain_qpoints()
    process.ctx.initialization = generate_hp_workchain_node()

    process.run_qpoints()
    # to keep consistency with QE we start from 1
    assert 'qpoint_1' in process.ctx
    assert 'qpoint_2' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_init(generate_workchain_qpoints, generate_hp_workchain_node):
    """Test `HpParallelizeQpointsWorkChain.inspect_init`."""
    process = generate_workchain_qpoints()
    process.ctx.initialization = generate_hp_workchain_node(exit_status=300)

    result = process.inspect_init()
    assert result == HpParallelizeQpointsWorkChain.exit_codes.ERROR_INITIALIZATION_WORKCHAIN_FAILED


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_qpoints(generate_workchain_qpoints, generate_hp_workchain_node):
    """Test `HpParallelizeQpointsWorkChain.inspect_qpoints`."""
    process = generate_workchain_qpoints()
    process.ctx.qpoint_1 = generate_hp_workchain_node(exit_status=300)

    result = process.inspect_qpoints()
    assert result == HpParallelizeQpointsWorkChain.exit_codes.ERROR_QPOINT_WORKCHAIN_FAILED


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_final(generate_workchain_qpoints, generate_hp_workchain_node):
    """Test `HpParallelizeQpointsWorkChain.inspect_final`."""
    process = generate_workchain_qpoints()
    process.ctx.compute_chi = generate_hp_workchain_node(exit_status=300)

    result = process.inspect_final()
    assert result == HpParallelizeQpointsWorkChain.exit_codes.ERROR_FINAL_WORKCHAIN_FAILED


@pytest.mark.usefixtures('aiida_profile')
def test_run_final(generate_workchain_qpoints, generate_hp_workchain_node):
    """Test `HpParallelizeQpointsWorkChain.run_final`."""
    process = generate_workchain_qpoints()
    process.ctx.qpoint_1 = generate_hp_workchain_node(use_retrieved=True)
    process.ctx.qpoint_2 = generate_hp_workchain_node(use_retrieved=True)

    process.run_final()

    assert 'compute_chi' in process.ctx
