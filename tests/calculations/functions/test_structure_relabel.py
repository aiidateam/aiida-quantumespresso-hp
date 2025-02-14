# -*- coding: utf-8 -*-
"""Test the :py:meth:`~aiida_hubbard.calculations.functions.structure_relabel_kinds` calcfunction."""
from aiida.orm import Dict
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

from aiida_hubbard.calculations.functions.structure_relabel_kinds import structure_relabel_kinds


def test_structure_relabel(generate_structure):
    """Test the `structure_relabel_kinds` calcfunction."""
    structure = generate_structure('AFMlicoo2')
    hubbard_structure = HubbardStructureData.from_structure(structure)
    hubbard_structure.initialize_onsites_hubbard('Co0', '3d')
    hubbard_structure.initialize_onsites_hubbard('Co1', '3d')
    hubbard_structure.initialize_onsites_hubbard('O', '2p')

    # We assume an AFM system where the Co sublattices need to be
    # again partitioned due to symmetry constraints (meaning they have different U).
    sites = [
        {
            'index': 0,
            'type': 1,
            'kind': 'Co0',
            'new_type': 1,
            'spin': 1
        },
        {
            'index': 1,
            'type': 1,
            'kind': 'Co0',
            'new_type': 2,
            'spin': 1
        },
        {
            'index': 2,
            'type': 3,
            'kind': 'Co1',
            'new_type': 3,
            'spin': -1
        },
        {
            'index': 3,
            'type': 3,
            'kind': 'Co1',
            'new_type': 4,
            'spin': -1
        },
        {
            'index': 4,
            'type': 5,
            'kind': 'O',
            'new_type': 5,
            'spin': 1
        },
        {
            'index': 5,
            'type': 5,
            'kind': 'O',
            'new_type': 5,
            'spin': 1
        },
    ]

    magnetization = Dict({'Co0': 0.5, 'Co1': -0.5})
    hubbard = Dict({'sites': sites})

    outputs = structure_relabel_kinds(hubbard_structure=hubbard_structure, hubbard=hubbard, magnetization=magnetization)

    relabeled = outputs['hubbard_structure']
    new_magnetization = outputs['starting_magnetization']

    assert relabeled.get_site_kindnames() == ['Co0', 'Co1', 'Co2', 'Co3', 'O0', 'O0', 'Li']
    assert new_magnetization.get_dict() == {'Co0': 0.5, 'Co1': 0.5, 'Co2': -0.5, 'Co3': -0.5}
