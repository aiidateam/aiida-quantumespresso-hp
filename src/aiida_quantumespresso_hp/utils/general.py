# -*- coding: utf-8 -*-
"""General utilies."""
from __future__ import annotations


def set_tot_magnetization(input_parameters: dict, tot_magnetization: float) -> bool:
    """Set the total magnetization based on its value and the input parameters.

    Set the `SYSTEM.tot_magnetization` parameters input equal to the round value of `tot_magnetization`.
    It returns whether the latter does not exceed within threshold from its original value.
    This is needed because `tot_magnetization` must be an integer for QuantumESPRESSO `pw.x`.
    """
    thr = 0.4  # threshold measuring the deviation from integer value

    int_tot_magnetization = round(tot_magnetization, 0)
    input_parameters['SYSTEM']['tot_magnetization'] = int_tot_magnetization

    return abs(tot_magnetization - int_tot_magnetization) < thr


def is_perturb_only_atom(parameters: dict) -> int | None:
    """Return the index of the ``perturb_only_atom`` key associated with the ``INPUTHP`` dictionary.

    :return: atomic index (QuantumESPRESSO format), None if the key is not in parameters
    """
    import re

    match = None  # making sure that if the dictionary is empty we don't raise an `UnboundLocalError`

    for key in parameters.keys():
        match = re.search(r'perturb_only_atom.*([0-9]).*', key)
        if match:
            if not parameters[key]:  # also the key must be `True`
                match = None  # making sure to have `None`
            else:
                match = int(match.group(1))
                break

    return match
