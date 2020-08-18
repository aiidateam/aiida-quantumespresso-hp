# -*- coding: utf-8 -*-
"""Tests for the `HpCalculation` class."""
import os

import pytest

from aiida.common import datastructures
from aiida.plugins import CalculationFactory

HpCalculation = CalculationFactory('quantumespresso.hp')


@pytest.mark.usefixtures('fixture_database')
def test_hp_default(fixture_sandbox_folder, generate_calc_job, generate_inputs_hp, file_regression):
    """Test a default `HpCalculation`."""
    entry_point_name = 'quantumespresso.hp'

    calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, generate_inputs_hp())

    filename_input = HpCalculation.spec().inputs.get_port('metadata.options.input_filename').default
    filename_output = HpCalculation.spec().inputs.get_port('metadata.options.output_filename').default

    cmdline_params = ['-in', filename_input]
    local_copy_list = []
    retrieve_list = []
    prefix = HpCalculation._prefix  # pylint: disable=protected-access

    retrieve_list.append(filename_output)
    retrieve_list.append(HpCalculation.filename_output_hubbard)
    retrieve_list.append(HpCalculation.filename_output_hubbard_parameters)
    retrieve_list.append(os.path.join(HpCalculation.dirname_output_hubbard, HpCalculation.filename_output_hubbard_chi))

    # Required files and directories for final collection calculations
    path_save_directory = os.path.join(HpCalculation.dirname_output, prefix + '.save')
    path_occup_file = os.path.join(HpCalculation.dirname_output, prefix + '.occup')
    path_paw_file = os.path.join(HpCalculation.dirname_output, prefix + '.paw')

    retrieve_list.append([path_save_directory, path_save_directory, 0])
    retrieve_list.append([path_occup_file, path_occup_file, 0])
    retrieve_list.append([path_paw_file, path_paw_file, 0])

    src_perturbation_files = os.path.join(HpCalculation.dirname_output_hubbard, '{}.chi.pert_*.dat'.format(prefix))
    dst_perturbation_files = '.'
    retrieve_list.append([src_perturbation_files, dst_perturbation_files, 3])

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    for element in calc_info.retrieve_list:
        assert element in retrieve_list

    with fixture_sandbox_folder.open(filename_input) as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox_folder.get_content_list()) == sorted([filename_input])
    file_regression.check(input_written, encoding='utf-8', extension='.in')
