# -*- coding: utf-8 -*-
# pylint: disable=unused-argument,protected-access
"""Tests for the `HpCalculation` class."""

import os

from aiida import orm
from aiida.common import datastructures
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.utils.resources import get_default_options

HpCalculation = CalculationFactory('quantumespresso.hp')


def test_hp_default(
    fixture_database, fixture_computer_localhost, fixture_sandbox_folder, generate_calc_job, generate_calc_job_node,
    generate_code_localhost, generate_kpoints_mesh, file_regression
):
    """Test a default `HpCalculation`."""
    entry_point_name = 'quantumespresso.hp'

    parameters = {'INPUTHP': {}}

    parent_inputs = {'parameters': orm.Dict(dict={'SYSTEM': {'lda_plus_u': True}})}
    parent_calculation = generate_calc_job_node('quantumespresso.pw', fixture_computer_localhost, inputs=parent_inputs)

    inputs = {
        'code': generate_code_localhost(entry_point_name, fixture_computer_localhost),
        'parent_folder': parent_calculation.outputs.remote_folder,
        'qpoints': generate_kpoints_mesh(2),
        'parameters': orm.Dict(dict=parameters),
        'metadata': {
            'options': get_default_options()
        }
    }

    calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, inputs)

    filename_input = HpCalculation.spec().inputs.get_port('metadata.options.input_filename').default
    filename_output = HpCalculation.spec().inputs.get_port('metadata.options.output_filename').default

    cmdline_params = ['-in', filename_input]
    local_copy_list = []
    retrieve_list = []

    retrieve_list.append(filename_output)
    retrieve_list.append(HpCalculation.filename_output_hubbard)
    retrieve_list.append(HpCalculation.filename_output_hubbard_chi)
    retrieve_list.append(HpCalculation.filename_output_hubbard_parameters)

    # Required files and directories for final collection calculations
    path_save_directory = os.path.join(HpCalculation._dirname_output, HpCalculation._prefix + '.save')
    path_occup_file = os.path.join(HpCalculation._dirname_output, HpCalculation._prefix + '.occup')
    path_paw_file = os.path.join(HpCalculation._dirname_output, HpCalculation._prefix + '.paw')

    retrieve_list.append([path_save_directory, path_save_directory, 0])
    retrieve_list.append([path_occup_file, path_occup_file, 0])
    retrieve_list.append([path_paw_file, path_paw_file, 0])

    src_perturbation_files = os.path.join(
        HpCalculation.dirname_output_hubbard, '{}.chi.pert_*.dat'.format(HpCalculation._prefix)
    )
    dst_perturbation_files = '.'
    retrieve_list.append([src_perturbation_files, dst_perturbation_files, 3])

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    # assert len(calc_info.retrieve_list) == len(retrieve_list)
    for element in calc_info.retrieve_list:
        assert element in retrieve_list

    with fixture_sandbox_folder.open(filename_input) as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox_folder.get_content_list()) == sorted([filename_input])
    file_regression.check(input_written, encoding='utf-8', extension='.in')
