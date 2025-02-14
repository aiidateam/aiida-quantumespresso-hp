# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `HpParallelizeAtomsWorkChain` class."""
from plumpy import ProcessState
import pytest

from aiida_hubbard.workflows.hp.parallelize_atoms import HpParallelizeAtomsWorkChain


@pytest.fixture
def generate_workchain_atoms(generate_workchain, generate_inputs_hp, generate_hubbard_structure):
    """Generate an instance of a `HpParallelizeAtomsWorkChain`."""

    def _generate_workchain_atoms(hp_inputs=None, parallelize_qpoints=False, max_concurrent_base_workchains=None):
        from aiida.orm import Bool, Int
        entry_point = 'quantumespresso.hp.parallelize_atoms'
        hp_inputs = generate_inputs_hp(inputs=hp_inputs)
        hp_inputs['hubbard_structure'] = generate_hubbard_structure()
        hp_inputs['parallelize_qpoints'] = Bool(parallelize_qpoints)
        inputs = {'hp': hp_inputs}
        if max_concurrent_base_workchains is not None:
            inputs['max_concurrent_base_workchains'] = Int(max_concurrent_base_workchains)
        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_atoms


@pytest.fixture
def generate_hp_workchain_node(generate_calc_job_node):
    """Generate an instance of `WorkflowNode`."""

    def _generate_hp_workchain_node(exit_status=0, use_retrieved=False):
        from aiida.common import LinkType
        from aiida.orm import Dict, WorkflowNode

        node = WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        parameters = Dict({
            'hubbard_sites': {
                '1': 'Co',
                '2': 'O',
            }
        }).store()
        parameters.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='parameters')

        if use_retrieved:
            retrieved = generate_calc_job_node(
                'quantumespresso.hp'
            ).outputs.retrieved  # otherwise the HpCalculation will complain
            retrieved.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='retrieved')

        return node

    return _generate_hp_workchain_node


@pytest.mark.usefixtures('aiida_profile')
def test_run_init(generate_workchain_atoms):
    """Test `HpParallelizeAtomsWorkChain.run_init`."""
    process = generate_workchain_atoms()
    process.run_init()

    assert 'initialization' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_atoms(generate_workchain_atoms, generate_hp_workchain_node):
    """Test `HpParallelizeAtomsWorkChain.run_atoms`."""
    process = generate_workchain_atoms()
    process.ctx.initialization = generate_hp_workchain_node()
    output_params = process.ctx.initialization.outputs.parameters.get_dict()
    process.ctx.hubbard_sites = list(output_params['hubbard_sites'].items())
    process.run_atoms()

    assert 'atom_1' in process.ctx
    assert 'atom_2' in process.ctx


@pytest.mark.usefixtures('aiida_profile')
def test_run_atoms_max_concurrent(generate_workchain_atoms, generate_hp_workchain_node):
    """Test `HpParallelizeAtomsWorkChain.run_atoms`.

    The number of concurrent `BaseWorkChains` is limited to `1`.
    """
    process = generate_workchain_atoms(max_concurrent_base_workchains=1)
    process.ctx.initialization = generate_hp_workchain_node()
    output_params = process.ctx.initialization.outputs.parameters.get_dict()
    process.ctx.hubbard_sites = list(output_params['hubbard_sites'].items())

    assert process.should_run_atoms()
    process.run_atoms()
    assert 'atom_1' in process.ctx
    assert 'atom_2' not in process.ctx
    assert process.should_run_atoms()
    process.run_atoms()
    assert 'atom_1' in process.ctx
    assert 'atom_2' in process.ctx

    assert not process.should_run_atoms()


@pytest.mark.usefixtures('aiida_profile')
def test_run_atoms_with_qpoints(generate_workchain_atoms, generate_hp_workchain_node):
    """Test `HpParallelizeAtomsWorkChain.run_atoms` with q point parallelization."""
    process = generate_workchain_atoms()
    process.ctx.initialization = generate_hp_workchain_node()
    output_params = process.ctx.initialization.outputs.parameters.get_dict()
    process.ctx.hubbard_sites = list(output_params['hubbard_sites'].items())
    process.run_atoms()

    # Don't know how to test something like the following
    # assert process.ctx.atom_1.__name__ == 'HpParallelizeQpointsWorkChain'


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_init(generate_workchain_atoms, generate_hp_workchain_node):
    """Test `HpParallelizeAtomsWorkChain.inspect_init`."""
    process = generate_workchain_atoms()
    process.ctx.initialization = generate_hp_workchain_node(exit_status=300)

    result = process.inspect_init()
    assert result == HpParallelizeAtomsWorkChain.exit_codes.ERROR_INITIALIZATION_WORKCHAIN_FAILED


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_atoms(generate_workchain_atoms, generate_hp_workchain_node):
    """Test `HpParallelizeAtomsWorkChain.inspect_atoms`."""
    process = generate_workchain_atoms()
    process.ctx.atom_1 = generate_hp_workchain_node(exit_status=300)

    result = process.inspect_atoms()
    assert result == HpParallelizeAtomsWorkChain.exit_codes.ERROR_ATOM_WORKCHAIN_FAILED


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_final(generate_workchain_atoms, generate_hp_workchain_node):
    """Test `HpParallelizeAtomsWorkChain.inspect_final`."""
    process = generate_workchain_atoms()
    process.ctx.compute_hp = generate_hp_workchain_node(exit_status=300)

    result = process.inspect_final()
    assert result == HpParallelizeAtomsWorkChain.exit_codes.ERROR_FINAL_WORKCHAIN_FAILED


@pytest.mark.usefixtures('aiida_profile')
def test_run_final(generate_workchain_atoms, generate_hp_workchain_node):
    """Test `HpParallelizeAtomsWorkChain.run_final`."""
    process = generate_workchain_atoms()
    process.ctx.atom_1 = generate_hp_workchain_node(use_retrieved=True)

    process.run_final()

    assert 'compute_hp' in process.ctx
