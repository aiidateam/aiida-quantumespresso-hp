# -*- coding: utf-8 -*-
# pylint: disable=no-member,redefined-outer-name
"""Tests for the `SelfConsistentHubbardWorkChain` class."""
from aiida.common import AttributeDict
from aiida.orm import Dict
from plumpy import ProcessState
import pytest


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


@pytest.fixture
def generate_scf_workchain_node(generate_hubbard_structure, generate_calc_job_node, generate_inputs_pw):
    """Generate an instance of `WorkflowNode`."""

    def _generate_scf_workchain_node(exit_status=0, relax=False, remote_folder=False):
        from aiida.common import LinkType
        from aiida.orm import WorkflowNode

        node = WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        parameters = Dict(dict={
            'number_of_bands': 1,
            'total_magnetization': 1,
        }).store()
        parameters.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='output_parameters')

        if relax:
            hubbard_structure = generate_hubbard_structure().store()
            hubbard_structure.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='output_structure')

        if remote_folder:
            inputs_pw = generate_inputs_pw()
            inputs_pw['structure'] = generate_hubbard_structure()
            remote_folder = generate_calc_job_node('quantumespresso.pw', inputs=inputs_pw).outputs.remote_folder
            remote_folder.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='remote_folder')

        return node

    return _generate_scf_workchain_node


@pytest.fixture
def generate_hp_workchain_node(generate_hubbard_structure):
    """Generate an instance of `WorkflowNode`."""

    def _generate_hp_workchain_node(exit_status=0, relabel=False, only_u=False, u_value=1e-5, v_value=1e-5):
        from aiida.common import LinkType
        from aiida.orm import WorkflowNode

        node = WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        hubbard_structure = generate_hubbard_structure(only_u=only_u, u_value=u_value, v_value=v_value).store()
        hubbard_structure.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='hubbard_structure')

        if relabel:
            sites = [
                {
                    'index': 0,
                    'type': 1,
                    'kind': 'Co',
                    'new_type': 2,
                    'spin': 1
                },
            ]
        else:
            sites = [
                {
                    'index': 0,
                    'type': 1,
                    'kind': 'Co',
                    'new_type': 1,
                    'spin': 1
                },
            ]

        hubbard = Dict({'sites': sites}).store()
        hubbard.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='hubbard')

        return node

    return _generate_hp_workchain_node


@pytest.mark.parametrize(('parameters', 'match'), (({
    'nspin': 2
}, r'Missing `starting_magnetization` input in `scf.pw.parameters` while `nspin == 2`.'), ({
    'nspin': 4
}, r'nspin=`.*` is not implemented in the `hp.x` code.')))
@pytest.mark.usefixtures('aiida_profile')
def test_validate_inputs_invalid_inputs(generate_workchain_hubbard, generate_inputs_hubbard, parameters, match):
    """Test `SelfConsistentHubbardWorkChain.validate_inputs` for invalid inputs."""
    inputs = AttributeDict(generate_inputs_hubbard())
    inputs.scf.pw.parameters['SYSTEM'].update(parameters)
    with pytest.raises(ValueError, match=match):
        generate_workchain_hubbard(inputs=inputs)


@pytest.mark.parametrize('parameters', ('skip_relax_iterations', 'relax_frequency'))
@pytest.mark.usefixtures('aiida_profile')
def test_validate_invalid_positve_input(generate_workchain_hubbard, generate_inputs_hubbard, parameters):
    """Test `SelfConsistentHubbardWorkChain` for invalid positive inputs."""
    from aiida.orm import Int

    inputs = AttributeDict(generate_inputs_hubbard())
    inputs.update({parameters: Int(-1)})

    match = 'the value must be positive.'
    with pytest.raises(ValueError, match=match):
        generate_workchain_hubbard(inputs=inputs)


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_hubbard, generate_inputs_hubbard):
    """Test `SelfConsistentHubbardWorkChain.setup`."""
    inputs = generate_inputs_hubbard()
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()

    assert process.ctx.iteration == 0
    assert process.ctx.relax_frequency == 1
    assert process.ctx.skip_relax_iterations == 0
    assert process.ctx.current_hubbard_structure == inputs['hubbard_structure']
    assert process.ctx.current_magnetic_moments is None
    assert not process.ctx.is_converged
    assert process.ctx.is_insulator is None
    assert not process.ctx.is_magnetic
    assert not process.should_check_convergence()


@pytest.mark.usefixtures('aiida_profile')
def test_reorder_atoms_setup(generate_workchain_hubbard, generate_inputs_hubbard, generate_structure):
    """Test `SelfConsistentHubbardWorkChain.setup` when reordering atoms."""
    from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

    structure = generate_structure(structure_id='licoo2')
    hubbard_structure = HubbardStructureData.from_structure(structure=structure)
    hubbard_structure.initialize_onsites_hubbard('O', '2p', 8.0)

    inputs = generate_inputs_hubbard()
    inputs['hubbard_structure'] = hubbard_structure
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()

    assert process.ctx.current_hubbard_structure != inputs['hubbard_structure']


@pytest.mark.usefixtures('aiida_profile')
def test_magnetic_setup(generate_workchain_hubbard, generate_inputs_hubbard):
    """Test `SelfConsistentHubbardWorkChain.setup` for magnetic systems."""
    inputs = AttributeDict(generate_inputs_hubbard())
    inputs.scf.pw.parameters['SYSTEM'].update({'nspin': 2, 'starting_magnetization': {'Co': 0.5}})
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()

    assert process.ctx.is_magnetic


@pytest.mark.usefixtures('aiida_profile')
def test_skip_relax_iterations(generate_workchain_hubbard, generate_inputs_hubbard, generate_hp_workchain_node):
    """Test `SelfConsistentHubbardWorkChain` when skipping the first relax iterations."""
    from aiida.orm import Bool, Int

    inputs = generate_inputs_hubbard()
    inputs['skip_relax_iterations'] = Int(1)
    inputs['meta_convergence'] = Bool(True)
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()
    # 1
    process.update_iteration()
    assert process.ctx.skip_relax_iterations == 1
    assert process.ctx.iteration == 1
    assert not process.should_run_relax()
    assert not process.should_check_convergence()
    process.ctx.workchains_hp = [generate_hp_workchain_node()]
    process.inspect_hp()
    assert process.ctx.current_hubbard_structure == process.ctx.workchains_hp[-1].outputs.hubbard_structure
    # 2
    process.update_iteration()
    assert process.should_run_relax()
    assert process.should_check_convergence()
    # 3
    process.update_iteration()
    assert process.should_run_relax()
    assert process.should_check_convergence()

    inputs['skip_relax_iterations'] = Int(2)
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()
    # 1
    process.update_iteration()
    assert process.ctx.skip_relax_iterations == 2
    assert not process.should_run_relax()
    assert not process.should_check_convergence()
    process.ctx.workchains_hp = [generate_hp_workchain_node()]
    process.inspect_hp()
    assert process.ctx.current_hubbard_structure == process.ctx.workchains_hp[-1].outputs.hubbard_structure
    # 2
    process.update_iteration()
    assert not process.should_run_relax()
    assert not process.should_check_convergence()
    process.ctx.workchains_hp.append(generate_hp_workchain_node())
    process.inspect_hp()
    assert process.ctx.current_hubbard_structure == process.ctx.workchains_hp[-1].outputs.hubbard_structure
    # 3
    process.update_iteration()
    assert process.should_run_relax()
    assert process.should_check_convergence()


@pytest.mark.usefixtures('aiida_profile')
def test_relax_frequency(generate_workchain_hubbard, generate_inputs_hubbard):
    """Test `SelfConsistentHubbardWorkChain` when `relax_frequency` is different from 1."""
    from aiida.orm import Int

    inputs = generate_inputs_hubbard()
    inputs['relax_frequency'] = Int(3)
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()

    process.update_iteration()
    assert not process.should_run_relax()  # skip
    process.update_iteration()
    assert not process.should_run_relax()  # skip
    process.update_iteration()
    assert process.should_run_relax()  # run
    process.update_iteration()
    assert not process.should_run_relax()  # skip


@pytest.mark.usefixtures('aiida_profile')
def test_radial_analysis(
    generate_workchain_hubbard,
    generate_inputs_hubbard,
    generate_scf_workchain_node,
):
    """Test `SelfConsistentHubbardWorkChain` outline when radial analysis is activated.

    We want to make sure `rmax` is in `hp.parameters`.
    """
    inputs = generate_inputs_hubbard()
    inputs['radial_analysis'] = Dict({})  # no need to specify inputs, it will use the defaults
    process = generate_workchain_hubbard(inputs=inputs)

    process.setup()
    process.ctx.workchains_scf = [generate_scf_workchain_node(remote_folder=True)]
    process.run_hp()

    # parameters = process.ctx['workchains_hp'][-1].inputs['hp']['parameters'].get_dict()
    # assert 'rmax' in parameters


@pytest.mark.usefixtures('aiida_profile')
def test_should_check_convergence(generate_workchain_hubbard, generate_inputs_hubbard):
    """Test `SelfConsistentHubbardWorkChain.should_check_convergence`."""
    from aiida.orm import Bool
    inputs = generate_inputs_hubbard()
    inputs['meta_convergence'] = Bool(True)
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()
    process.update_iteration()
    assert process.should_check_convergence()


@pytest.mark.usefixtures('aiida_profile')
def test_outline_without_metaconvergence(
    generate_workchain_hubbard, generate_inputs_hubbard, generate_hp_workchain_node
):
    """Test `SelfConsistentHubbardWorkChain` outline without metaconvergece.

    We want to make sure the `outputs.hubbard_structure` is the last computed.
    """
    from aiida.orm import Bool
    inputs = generate_inputs_hubbard()
    inputs['meta_convergence'] = Bool(False)
    process = generate_workchain_hubbard(inputs=inputs)

    process.setup()

    process.ctx.workchains_hp = [generate_hp_workchain_node()]
    assert process.inspect_hp() is None
    assert process.ctx.is_converged

    process.run_results()
    assert 'hubbard_structure' in process.outputs
    assert process.outputs['hubbard_structure'] == process.ctx.workchains_hp[-1].outputs['hubbard_structure']


@pytest.mark.usefixtures('aiida_profile')
def test_outline(
    generate_workchain_hubbard, generate_inputs_hubbard, generate_scf_workchain_node, generate_hp_workchain_node
):
    """Test `SelfConsistentHubbardWorkChain` outline."""
    from aiida.orm import Bool
    inputs = generate_inputs_hubbard()
    inputs['meta_convergence'] = Bool(True)
    process = generate_workchain_hubbard(inputs=inputs)

    process.setup()

    process.run_relax()
    # assert 'workchains_relax' in process.ctx
    # assert len(process.ctx.workchains_relax) == 1

    # Mock the `workchains_scf` context variable as if a `PwRelaxWorkChain` has been run in
    process.ctx.workchains_relax = [generate_scf_workchain_node(relax=True)]
    result = process.inspect_relax()
    assert result is None
    assert process.ctx.current_hubbard_structure == process.ctx.workchains_relax[-1].outputs.output_structure

    process.run_scf_smearing()
    # assert 'workchains_scf' in process.ctx
    # assert len(process.ctx.workchains_scf) == 1

    # Mock the `workchains_scf` context variable as if a `PwBaseWorkChain` has been run in
    process.ctx.workchains_scf = [generate_scf_workchain_node(remote_folder=True)]
    process.run_scf_fixed()
    # assert len(process.ctx.workchains_scf) == 2

    # Mock the `workchains_scf` context variable as if a `PwBaseWorkChain` has been run in
    process.ctx.workchains_scf = [generate_scf_workchain_node(remote_folder=True)]
    process.run_hp()
    # assert 'workchains_hp' in process.ctx
    # assert len(process.ctx.workchains_hp) == 1

    process.ctx.workchains_hp = [generate_hp_workchain_node()]
    assert process.inspect_hp() is None
    process.check_convergence()
    assert process.ctx.is_converged

    process.run_results()
    assert 'hubbard_structure' in process.outputs
    assert process.outputs['hubbard_structure'] == process.ctx.workchains_hp[-1].outputs['hubbard_structure']


@pytest.mark.usefixtures('aiida_profile')
def test_should_run_relax(generate_workchain_hubbard, generate_inputs_hubbard):
    """Test `SelfConsistentHubbardWorkChain.should_run_relax` method."""
    from aiida.orm import Bool
    inputs = generate_inputs_hubbard()
    inputs['meta_convergence'] = Bool(True)
    inputs.pop('relax')
    process = generate_workchain_hubbard(inputs=inputs)

    process.setup()

    assert not process.should_run_relax()


@pytest.mark.usefixtures('aiida_profile')
def test_converged_check_convergence(
    generate_workchain_hubbard, generate_hp_workchain_node, generate_inputs_hubbard, generate_hubbard_structure
):
    """Test when `SelfConsistentHubbardWorkChain.check_convergence` is at convergence."""
    inputs = generate_inputs_hubbard()
    process = generate_workchain_hubbard(inputs=inputs)

    process.setup()

    # Mocking current (i.e. "old") and "new" HubbardStructureData,
    # containing different Hubbard parameters
    process.ctx.current_hubbard_structure = generate_hubbard_structure(only_u=True)
    process.ctx.workchains_hp = [generate_hp_workchain_node(only_u=True)]
    process.check_convergence()

    assert process.ctx.is_converged

    process.ctx.current_hubbard_structure = generate_hubbard_structure()
    process.ctx.workchains_hp = [generate_hp_workchain_node()]

    process.check_convergence()
    assert process.ctx.is_converged


@pytest.mark.usefixtures('aiida_profile')
def test_not_converged_check_convergence(
    generate_workchain_hubbard, generate_hp_workchain_node, generate_inputs_hubbard, generate_hubbard_structure
):
    """Test when `SelfConsistentHubbardWorkChain.check_convergence` is not at convergence."""
    inputs = generate_inputs_hubbard()
    process = generate_workchain_hubbard(inputs=inputs)

    process.setup()

    process.ctx.current_hubbard_structure = generate_hubbard_structure()
    process.ctx.workchains_hp = [generate_hp_workchain_node(u_value=5.0)]

    process.check_convergence()
    assert not process.ctx.is_converged

    process.ctx.current_hubbard_structure = generate_hubbard_structure()
    process.ctx.workchains_hp = [generate_hp_workchain_node(v_value=1.0)]

    process.check_convergence()
    assert not process.ctx.is_converged


@pytest.mark.usefixtures('aiida_profile')
def test_relabel_check_convergence(
    generate_workchain_hubbard, generate_hp_workchain_node, generate_inputs_hubbard, generate_hubbard_structure
):
    """Test when `SelfConsistentHubbardWorkChain.check_convergence` when relabelling is needed."""
    inputs = generate_inputs_hubbard()
    process = generate_workchain_hubbard(inputs=inputs)

    process.setup()

    current_hubbard_structure = generate_hubbard_structure(u_value=1, only_u=True)
    process.ctx.current_hubbard_structure = current_hubbard_structure
    process.ctx.workchains_hp = [generate_hp_workchain_node(relabel=True, u_value=100, only_u=True)]
    process.check_convergence()
    assert not process.ctx.is_converged
    assert process.ctx.current_hubbard_structure.get_kind_names() != current_hubbard_structure.get_kind_names()

    current_hubbard_structure = generate_hubbard_structure(u_value=99.99, only_u=True)
    process.ctx.current_hubbard_structure = current_hubbard_structure
    process.ctx.workchains_hp = [generate_hp_workchain_node(relabel=True, u_value=100, only_u=True)]
    process.check_convergence()
    assert process.ctx.is_converged
    assert process.ctx.current_hubbard_structure.get_kind_names() != current_hubbard_structure.get_kind_names()

    current_hubbard_structure = generate_hubbard_structure(u_value=99.99)
    process.ctx.current_hubbard_structure = current_hubbard_structure
    process.ctx.workchains_hp = [generate_hp_workchain_node(relabel=True, u_value=100)]
    process.check_convergence()
    assert process.ctx.is_converged
    assert process.ctx.current_hubbard_structure.get_kind_names() == current_hubbard_structure.get_kind_names()


@pytest.mark.usefixtures('aiida_profile')
def test_inspect_hp(generate_workchain_hubbard, generate_inputs_hubbard, generate_hp_workchain_node):
    """Test `SelfConsistentHubbardWorkChain.inspect_hp`."""
    from aiida_quantumespresso_hp.workflows.hubbard import SelfConsistentHubbardWorkChain as WorkChain
    inputs = generate_inputs_hubbard()
    process = generate_workchain_hubbard(inputs=inputs)
    process.setup()
    process.ctx.workchains_hp = [generate_hp_workchain_node(exit_status=300)]
    result = process.inspect_hp()
    assert result == WorkChain.exit_codes.ERROR_SUB_PROCESS_FAILED_HP.format(iteration=process.ctx.iteration)
