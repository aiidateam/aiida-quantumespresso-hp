# -*- coding: utf-8 -*-
"""Test the automatic ``invalidates_cache`` attribute for exit codes."""
import inspect

from aiida.engine import CalcJob
from aiida.plugins import CalculationFactory
import pytest


@pytest.mark.parametrize('entry_point_name', ['quantumespresso.hp'])
def test_exit_code_invalidates_cache(entry_point_name):
    """Test automatic ``invalidates_cache`` attribute of exit codes.

    Test that the ``invalidates_cache`` attribute of exit codes is automatically set according to the status integer.
    """
    entry_point = CalculationFactory(entry_point_name)

    if not inspect.isclass(entry_point) or not issubclass(entry_point, CalcJob):
        return

    for exit_code in entry_point.exit_codes.values():
        if exit_code.status < 400:
            assert exit_code.invalidates_cache
        else:
            assert not exit_code.invalidates_cache
