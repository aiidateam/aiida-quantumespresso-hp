# -*- coding: utf-8 -*-
from aiida.orm import CalculationFactory


PwCalculation = CalculationFactory('quantumespresso.pw')


def validate_parent_calculation(calculation):
    """
    Validate whether the given calculation is a valid parent calculation for a HpCalculation

    :param calculation: the calculation to validate
    :raises ValueError: if the calculation is not a valid parent calculation for a HpCalculation
    """
    if not isinstance(calculation, PwCalculation):
        raise ValueError('the parent calculation should be of type PwCalculation')

    try:
        parameters = calculation.inp.parameters.get_dict()
    except KeyError:
        raise ValueError('could not retrieve the input parameters node')

    lda_plus_u = parameters.get('SYSTEM', {}).get('lda_plus_u', None)
    hubbard_u = parameters.get('SYSTEM', {}).get('hubbard_u', {})

    if lda_plus_u is not True:
        print lda_plus_u
        raise ValueError('the parent calculation did not set lda_plus_u = True')

    if not hubbard_u:
        raise ValueError('the parent calculation did not specify any Hubbard kinds')

    try:
        structure = calculation.inp.structure
    except KeyError:
        raise ValueError('could not retrieve the input structure node')

    validate_structure_kind_order(structure, hubbard_u.keys())


def validate_structure_kind_order(structure, hubbard_kinds):
    """
    Determines whether the kinds in the structure node have the right order for the given list
    of Hubbard U kinds. For the order to be right, means for the Hubbard kinds to come first in
    the list of kinds of the structure

    :param structure: StructureData node
    :param hubbard_kinds: a list of Hubbard kinds
    """
    for kind in structure.kinds:

        if not hubbard_kinds:
            return
        elif kind.name in hubbard_kinds:
            hubbard_kinds.remove(kind.name)
        else:
            raise ValueError('the structure does not have the right kind order')