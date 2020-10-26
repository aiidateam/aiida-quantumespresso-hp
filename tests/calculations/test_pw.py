# -*- coding: utf-8 -*-
"""Tests for the ``PwCalculation`` class.

The ``PwCalculation`` class, although part of the ``aiida-quantumespresso`` plugin, is the calculation plugin that is
responsible for taking input that is specific to Hubbard parameter calculations.
"""
import io

import pytest

from aiida import orm
from aiida.plugins import CalculationFactory

HpCalculation = CalculationFactory('quantumespresso.hp')


@pytest.mark.usefixtures('fixture_database')
def test_hubbard_file(fixture_sandbox_folder, generate_calc_job, generate_inputs_pw):
    """Test a ``PwCalculation`` passing the ``hubbard_file`` as input."""
    entry_point_name = 'quantumespresso.pw'

    hubbard_file = orm.SinglefileData(io.BytesIO(b'content'), filename='parameters.in')
    inputs = {
        'hubbard_file': hubbard_file,
    }
    calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, generate_inputs_pw(**inputs))
    assert (
        hubbard_file.uuid, hubbard_file.filename, HpCalculation.filename_input_hubbard_parameters
    ) in calc_info.local_copy_list
