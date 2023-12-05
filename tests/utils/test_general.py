# -*- coding: utf-8 -*-
"""Tests for the :mod:`aiida_quantumespresso_hp.utils.general` module."""


def test_set_tot_magnetization():
    """Test the `set_tot_magnetization` function."""
    from aiida_quantumespresso_hp.utils.general import set_tot_magnetization

    parameters = {'SYSTEM': {}}

    assert set_tot_magnetization(parameters, 0.1)
    assert parameters['SYSTEM']['tot_magnetization'] == 0

    parameters = {'SYSTEM': {}}

    assert not set_tot_magnetization(parameters, 0.5)


def test_is_perturb_only_atom():
    """Test the `is_perturb_only_atom` function."""
    from aiida_quantumespresso_hp.utils.general import is_perturb_only_atom

    parameters = {}
    assert is_perturb_only_atom(parameters) is None

    parameters = {'perturb_only_atom(1)': True}
    assert is_perturb_only_atom(parameters) == 1

    parameters = {'perturb_only_atom(20)': True}
    assert is_perturb_only_atom(parameters) == 20

    parameters = {'perturb_only_atom(1)': False}
    assert is_perturb_only_atom(parameters) is None
