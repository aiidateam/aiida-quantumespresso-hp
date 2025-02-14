# -*- coding: utf-8 -*-
"""Calculation function to relabel the kinds of a Hubbard structure."""
from __future__ import annotations

from copy import deepcopy

from aiida.engine import calcfunction
from aiida.orm import Dict
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData


@calcfunction
def structure_relabel_kinds(
    hubbard_structure: HubbardStructureData,
    hubbard: Dict,
    magnetization: dict | None = None,
) -> Dict:
    """Create a clone of the given structure but with new kinds, based on the new hubbard sites.

    :param hubbard_structure: ``HubbardStructureData`` instance.
    :param hubbard: the ``hubbard`` output Dict node of a ``HpCalculation``.
    :param magnetization: Dict instance containing the `starting_magnetization` QuantumESPRESSO inputs.
    :return: dict with keys:

        * ``hubbard_structure``: relabelled ``HubbardStructureData``
        * ``starting_magnetization``: updated magnetization as :class:`~aiida.orm.Dict` (if provided in inputs)

    """
    relabeled = hubbard_structure.clone()
    relabeled.clear_kinds()
    relabeled.clear_sites()
    type_to_kind = {}
    sites = hubbard_structure.sites

    if magnetization:
        old_magnetization = magnetization.get_dict()
        new_magnetization = deepcopy(old_magnetization)
        # Removing old Hubbard spin-polarized atom label.
        for site in hubbard['sites']:
            new_magnetization.pop(site['kind'], None)

    kind_set = hubbard_structure.get_site_kindnames()
    symbol_set = [hubbard_structure.get_kind(kind_name).symbol for kind_name in kind_set]
    symbol_counter = {key: 0 for key in hubbard_structure.get_symbols_set()}

    # First do the Hubbard sites, popping the kind name suffix each time a new type is encountered. We do the suffix
    # generation ourselves, because the indexing done by hp.x contains gaps in the sequence.
    for index, site in enumerate(hubbard['sites']):
        symbol = symbol_set[index]

        try:
            # We define a `spin_type`, since ``hp.x`` does not distinguish new types according to spin
            spin_type = str(int(site['new_type']) * int(site['spin']))
            kind_name = type_to_kind[spin_type]
        except KeyError:
            kind_name = get_relabelled_symbol(symbol, symbol_counter[symbol])
            type_to_kind[spin_type] = kind_name
            symbol_counter[symbol] += 1

        if magnetization:
            # filling 'starting magnetization' with input value but new label;
            # if does not present a starting value, pass.
            if site['kind'] in old_magnetization:
                new_magnetization[kind_name] = old_magnetization[site['kind']]

        site = sites[index]
        try:
            relabeled.append_atom(position=site.position, symbols=symbol, name=kind_name)
        except ValueError as exc:
            raise ValueError('cannot distinguish kinds with the given Hubbard input configuration') from exc

    # Now add the non-Hubbard sites
    for site in sites[len(relabeled.sites):]:
        symbols = hubbard_structure.get_kind(site.kind_name).symbols
        names = hubbard_structure.get_kind(site.kind_name).name
        relabeled.append_atom(position=site.position, symbols=symbols, name=names)

    outputs = {'hubbard_structure': relabeled}
    if magnetization:
        outputs.update({'starting_magnetization': Dict(new_magnetization)})

    return outputs


def get_relabelled_symbol(symbol: str, counter: int) -> str:
    """Return a relabelled symbol.

    .. warning:: this function produces up to 36 different chemical symbols.

    :param symbol: a chemical symbol, NOT a kind name
    :param counter: a integer to assing the new label. Up to 9 an interger
        is appended, while an *ascii uppercase letter* is used. Lower cases
        are discarded to avoid possible misleading names
    :return: a 3 digit length symbol (QuantumESPRESSO allows only up to 3)
    """
    from string import ascii_uppercase, digits
    suffix = (digits + ascii_uppercase)[counter]
    return f'{symbol}{suffix}'
