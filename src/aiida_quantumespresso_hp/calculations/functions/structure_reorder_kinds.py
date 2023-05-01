# -*- coding: utf-8 -*-
"""Calculation function to reorder the kinds of a structure with the Hubbard sites first."""
from aiida.engine import calcfunction
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData
from aiida_quantumespresso.utils.hubbard import HubbardUtils


@calcfunction
def structure_reorder_kinds(hubbard_structure: HubbardStructureData) -> HubbardStructureData:
    """Create a copy of the structure but with the kinds in the right order necessary for an ``hp.x`` calculation.

    An ``HpCalculation`` which restarts from a completed ``PwCalculation``,
    requires that the all Hubbard atoms appear first in
    the atomic positions card of the PwCalculation input file.
    This order is based on the order of the kinds in the
    structure. So a correct structure has all Hubbard kinds
    in the begining of kinds list.

    :param hubbard_structure: reordered :class:`aiida_quantumespresso.data.hubbard.HubbardStructureData` node
    """
    reordered = hubbard_structure.clone()

    hubbard_utils = HubbardUtils(reordered)
    hubbard_utils.reorder_atoms()

    return hubbard_utils.hubbard_structure
