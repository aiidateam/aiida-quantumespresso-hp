# -*- coding: utf-8 -*-
"""Calculation function to reorder the kinds of a structure with the Hubbard sites first."""
from copy import deepcopy

from aiida.engine import calcfunction


@calcfunction
def structure_reorder_kinds(structure, hubbard_u):
    """Create a copy of the structure but with the kinds in the right order necessary for an hp.x calculation

    An HpCalculation which restarts from a completed PwCalculation, requires that the all Hubbard atoms appear first in
    the atomic positions card of the PwCalculation input file. This order is based on the order of the kinds in the
    structure. So a correct structure has all Hubbard kinds in the begining of kinds list.

    :param structure: StructureData node
    :param hubbard_u: a Dict node with the Hubbard U kinds and their values
    """
    reordered = deepcopy(structure)
    reordered.clear_kinds()

    sites = structure.sites
    hubbard_kinds = list(hubbard_u.get_dict().keys())
    hubbard_kinds.sort(reverse=True)

    ordered_sites = []

    while hubbard_kinds:

        hubbard_kind = hubbard_kinds.pop()

        hubbard_sites = []
        remaining_sites = []

        hubbard_sites = [s for s in sites if s.kind_name == hubbard_kind]
        remaining_sites = [s for s in sites if not s.kind_name == hubbard_kind]

        ordered_sites.extend(hubbard_sites)
        sites = remaining_sites

    # Extend the current site list with the remaining non-hubbard sites
    ordered_sites.extend(sites)

    for site in ordered_sites:

        if site.kind_name not in reordered.get_kind_names():
            kind = structure.get_kind(site.kind_name)
            reordered.append_kind(kind)

        reordered.append_site(site)

    return reordered
