# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `HpWorkChain` class."""
from plumpy import ProcessState
import pytest

from aiida_hubbard.workflows.hp.main import HpWorkChain


@pytest.fixture
def generate_workchain_main(generate_workchain, generate_inputs_hp, generate_hubbard_structure):
    """Generate an instance of a `HpWorkChain`."""

    def _generate_workchain_main(inputs=None, atoms=True, qpoints=True, qdistance=True):
        from aiida.orm import Bool, Float

        entry_point = 'quantumespresso.hp.main'

        inputs = generate_inputs_hp(inputs=inputs)
        inputs['hubbard_structure'] = generate_hubbard_structure()

        workchain_inputs = {
            'hp': inputs,
            'parallelize_atoms': Bool(atoms),
            'parallelize_qpoints': Bool(qpoints),
        }

        if qdistance:
            workchain_inputs['qpoints_distance'] = Float(0.15)
        else:
            workchain_inputs['qpoints'] = workchain_inputs['hp'].pop('qpoints')

        process = generate_workchain(entry_point, workchain_inputs)

        return process

    return _generate_workchain_main


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


def test_validate_inputs_invalid(generate_workchain_main):
    """Test `HpWorkChain.validate_inputs` with invalid inputs."""
    match = r'To use `parallelize_qpoints`, also `parallelize_atoms` must be `True`'
    with pytest.raises(ValueError, match=match):
        generate_workchain_main(atoms=False, qpoints=True)


@pytest.mark.usefixtures('aiida_profile')
def test_validate_qpoints(generate_workchain_main):
    """Test `HpWorkChain.validate_qpoints`."""
    process = generate_workchain_main()
    process.validate_qpoints()
    assert 'qpoints' in process.ctx

    process = generate_workchain_main(qdistance=True)
    assert process.validate_qpoints() is None
    assert 'qpoints' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_should_parallelize_atoms(generate_workchain_main):
    """Test `HpWorkChain.should_parallelize_atoms`."""
    process = generate_workchain_main()
    process.validate_qpoints()
    assert process.should_parallelize_atoms()


@pytest.mark.usefixtures('aiida_profile')
def test_run_base_workchain(generate_workchain_main):
    """Test `HpWorkChain.run_base_workchain`."""
    process = generate_workchain_main()
    process.validate_qpoints()
    result = process.run_base_workchain()
    assert result is not None


@pytest.mark.usefixtures('aiida_profile')
def test_run_parallel_workchain(generate_workchain_main):
    """Test `HpWorkChain.run_parallel_workchain`."""
    process = generate_workchain_main()
    process.validate_qpoints()
    result = process.run_parallel_workchain()
    assert result is not None


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_workchain(generate_workchain_main, generate_hp_workchain_node):
    """Test `HpWorkChain.inspect_workchain`."""
    process = generate_workchain_main()
    process.validate_qpoints()
    process.ctx.workchain = generate_hp_workchain_node(exit_status=300)

    result = process.inspect_workchain()
    assert result == HpWorkChain.exit_codes.ERROR_CHILD_WORKCHAIN_FAILED
