# -*- coding: utf-8 -*-
"""Tests for the `HpCalculation` class."""
import os

from aiida import orm
from aiida.common import datastructures
from aiida.plugins import CalculationFactory
import pytest

HpCalculation = CalculationFactory('quantumespresso.hp')


def test_default(fixture_sandbox_folder, generate_calc_job, generate_inputs_hp, file_regression):
    """Test a default `HpCalculation`."""
    entry_point_name = 'quantumespresso.hp'

    calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, generate_inputs_hp())

    filename_input = HpCalculation.spec().inputs.get_port('metadata.options.input_filename').default
    filename_output = HpCalculation.spec().inputs.get_port('metadata.options.output_filename').default

    cmdline_params = ['-in', filename_input]
    retrieve_list = []
    prefix = HpCalculation.prefix

    retrieve_list.append(filename_output)
    retrieve_list.append(HpCalculation.filename_output_hubbard)
    retrieve_list.append(HpCalculation.filename_output_hubbard_dat)
    retrieve_list.append(os.path.join(HpCalculation.dirname_output_hubbard, HpCalculation.filename_output_hubbard_chi))

    src_perturbation_files = os.path.join(HpCalculation.dirname_output_hubbard, f'{prefix}.*.pert_*.dat')
    dst_perturbation_files = '.'
    retrieve_list.append((src_perturbation_files, dst_perturbation_files, 3))

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert calc_info.local_copy_list is None
    for element in calc_info.retrieve_list:
        assert element in retrieve_list

    with fixture_sandbox_folder.open(filename_input) as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox_folder.get_content_list()) == sorted([filename_input])
    file_regression.check(input_written, encoding='utf-8', extension='.in')


def test_settings(fixture_sandbox_folder, generate_calc_job, generate_inputs_hp):
    """Test a default `HpCalculation` with `settings` in inputs."""
    entry_point_name = 'quantumespresso.hp'

    inputs = generate_inputs_hp()
    cmdline_params = ['-nk', '4', '-nband', '2', '-ntg', '3', '-ndiag', '12']
    inputs['settings'] = orm.Dict({'cmdline': cmdline_params})
    calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, inputs)

    # Check that the command-line parameters are as expected.
    assert calc_info.codes_info[0].cmdline_params == cmdline_params + ['-in', 'aiida.in']


@pytest.mark.parametrize(('parameters', 'match'), (
    ({
        'nq1': 1
    }, r'explicit definition of flag `nq1` in namelist `.*` is not allowed'),
    (
        {
            'compute_hp': True,
            'use_parent_hp': False,
            'use_hubbard_structure': False
        },
        (
            r'parameter `INPUTHP.compute_hp` is `True` but no parent folders '
            r'defined in `parent_hp` or no `hubbard_structure` in inputs'
        ),
    ),
    (
        {
            'determine_q_mesh_only': True
        },
        r'parameter `INPUTHP.determine_q_mesh_only` is `True` but `INPUTHP.perturb_only_atom` is not set',
    ),
    ({
        'determine_q_mesh_only': True,
        'determine_num_pert_only': True,
        'use_hubbard_structure': True
    }, r'parameter `INPUTHP.determine_q_mesh_only` is `True` but `INPUTHP.determine_num_pert_only` is `True` as well'),
))
def test_invalid_parameters(
    fixture_sandbox_folder,
    generate_calc_job,
    generate_calc_job_node,
    generate_hubbard_structure,
    generate_inputs_hp,
    parameters,
    match,
):
    """Test validation of `parameters`."""
    use_hubbard_structure = parameters.pop('use_hubbard_structure', True)
    use_parent_hp = parameters.pop('use_parent_hp', True)

    inputs = generate_inputs_hp(inputs=parameters)

    if use_hubbard_structure:
        inputs['hubbard_structure'] = generate_hubbard_structure()

    if use_parent_hp:
        inputs['parent_hp'] = {'site_01': generate_calc_job_node('quantumespresso.hp').outputs.retrieved}

    with pytest.raises(ValueError, match=match):
        generate_calc_job(fixture_sandbox_folder, 'quantumespresso.hp', inputs)


def test_invalid_qpoints(fixture_sandbox_folder, generate_calc_job, generate_inputs_hp):
    """Test validation of `qpoints`."""
    qpoints = orm.KpointsData()
    qpoints.set_kpoints_mesh([2, 2, 2], [0.5, 0.5, 0.5])

    inputs = generate_inputs_hp()
    inputs['qpoints'] = qpoints

    with pytest.raises(ValueError, match=r'support for qpoint meshes with non-zero offsets is not implemented'):
        generate_calc_job(fixture_sandbox_folder, 'quantumespresso.hp', inputs)


def test_invalid_parent_scf(
    fixture_sandbox_folder, generate_calc_job, generate_inputs_hp, generate_calc_job_node, generate_structure
):
    """Test validation of `parent_scf`."""
    inputs = generate_inputs_hp()
    inputs_pw = {'structure': generate_structure()}

    inputs['parent_scf'] = generate_calc_job_node(
        'quantumespresso.hp', inputs=inputs_pw
    ).outputs.remote_folder  # note the .hp
    with pytest.raises(ValueError, match=r'creator of `parent_scf` .* is not a `PwCalculation`'):
        generate_calc_job(fixture_sandbox_folder, 'quantumespresso.hp', inputs)

    inputs['parent_scf'] = generate_calc_job_node('quantumespresso.pw', inputs=inputs_pw).outputs.remote_folder
    with pytest.raises(ValueError, match=r'parent calculation .* was not run with `HubbardStructureData`'):
        generate_calc_job(fixture_sandbox_folder, 'quantumespresso.hp', inputs)


def test_invalid_parent_hp(fixture_sandbox_folder, generate_calc_job, generate_inputs_hp, generate_calc_job_node):
    """Test validation of `parent_hp`."""
    inputs = generate_inputs_hp()

    inputs['parent_hp'] = {'site_01': generate_calc_job_node('quantumespresso.pw').outputs.retrieved}
    with pytest.raises(ValueError, match=r'creator of `parent_hp.site_01` .* is not a `HpCalculation`'):
        generate_calc_job(fixture_sandbox_folder, 'quantumespresso.hp', inputs)


def test_collect_no_parents(fixture_sandbox_folder, generate_calc_job, generate_inputs_hp):
    """Test a `HpCalculation` performing a `compute_hp` calculation but without parent folder specified."""
    inputs = generate_inputs_hp(inputs={'compute_hp': True})

    with pytest.raises(ValueError, match=r'.*`INPUTHP.compute_hp` is `True` but no parent folders defined.*'):
        generate_calc_job(fixture_sandbox_folder, 'quantumespresso.hp', inputs)


def test_collect(fixture_sandbox_folder, generate_calc_job, generate_inputs_hp, generate_hp_retrieved, file_regression):
    """Test a `HpCalculation` performing a `compute_hp` calculation."""
    entry_point_name = 'quantumespresso.hp'
    filename_input = HpCalculation.spec().inputs.get_port('metadata.options.input_filename').default
    inputs = generate_inputs_hp()
    inputs['parent_hp'] = {'site_01': generate_hp_retrieved}

    calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, inputs)

    assert calc_info.provenance_exclude_list

    with fixture_sandbox_folder.open(filename_input) as handle:
        input_written = handle.read()

    assert sorted(fixture_sandbox_folder.get_content_list()) == sorted([filename_input])
    file_regression.check(input_written, encoding='utf-8', extension='.in')
