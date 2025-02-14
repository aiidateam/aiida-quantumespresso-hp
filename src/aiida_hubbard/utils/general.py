# -*- coding: utf-8 -*-
"""General utilies."""
from __future__ import annotations

from typing import List


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
        match = re.search(r'perturb_only_atom.*?(\d+).*', key)
        if match:
            if not parameters[key]:  # also the key must be `True`
                match = None  # making sure to have `None`
            else:
                match = int(match.group(1))
                break

    return match


def distribute_base_workchains(n_atoms: int, n_total: int) -> List[int]:
    """Distribute the maximum number of `BaseWorkChains` to be launched.

    The number of `BaseWorkChains` will be distributed over the number of atoms.
    The elements of the resulting list correspond to the number of q-point
    `BaseWorkChains` to be launched for each atom, in case q-point parallelization
    is used. Otherwise, the method will only take care of limitting the number
    of `HpParallelizeAtomsWorkChain` to be launched in parallel.

    :param n_atoms: The number of atoms.
    :param n_total: The number of base workchains to be launched.
    :return: The number of base workchains to be launched for each atom.
    """
    quotient = n_total // n_atoms
    remainder = n_total % n_atoms
    n_distributed = [quotient] * n_atoms

    for i in range(remainder):
        n_distributed[i] += 1

    n_distributed = [x for x in n_distributed if x != 0]

    return n_distributed
