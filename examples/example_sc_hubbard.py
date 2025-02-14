#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""Test actual run of self-consistent Hubbard workchain."""
from aiida import load_profile
from aiida.engine import run
from aiida.orm import StructureData, load_code
from aiida_quantumespresso.common.types import ElectronicType, SpinType  # RelaxType
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

from aiida_hubbard.workflows.hubbard import SelfConsistentHubbardWorkChain

load_profile()


def test_self_consistent_hubbard():
    """Run a simple example of the self-consistent Hubbard workflow."""
    # LiCoO2 structure used in QuantumESPRESSO HP examples.
    a, b, c, d = 1.40803, 0.81293, 4.68453, 1.62585
    cell = [[a, -b, c], [0.0, d, c], [-a, -b, c]]
    positions = [[0, 0, 0], [0, 0, 3.6608], [0, 0, 10.392], [0, 0, 7.0268]]
    symbols = ['Co', 'O', 'O', 'Li']
    structure = StructureData(cell=cell)
    for position, symbol in zip(positions, symbols):
        structure.append_atom(position=position, symbols=symbol)

    hubbard_structure = HubbardStructureData.from_structure(structure)
    hubbard_structure.initialize_onsites_hubbard('Co', '3d', 5.0)

    pw_code = load_code('pw@localhost')
    hp_code = load_code('hp@localhost')

    kwargs = {
        'electronic_type': ElectronicType.INSULATOR,
        # 'relax_type': RelaxType.POSITIONS, # uncomment this to relax positions only, and related lines below
        'spin_type': SpinType.NONE,
    }

    builder = SelfConsistentHubbardWorkChain.get_builder_from_protocol(
        pw_code=pw_code,
        hp_code=hp_code,
        hubbard_structure=hubbard_structure,
        protocol='fast',
        overrides={ # this can be more conveniently moved on file as .yaml file
            # 'relax':{
            #     'base':{
            #         'kpoints_distance': 100.0, # so high that it gives 1x1x1
            #     },
            # },
            'scf':{
                'kpoints_distance': 100.0, # so high that it gives 1x1x1
                'pw':{
                    'parameters':{
                        'SYSTEM':{
                            'ecutwfc': 30.0,
                            'ecutrho': 30.0 * 8,
                        },
                    },
                },
            },
            'hubbard':{
                'parallelize_atoms': True,
                'parallelize_qpoints': True,
                'qpoints_distance': 100.0, # so high that it gives 1x1x1
            },
        },
        **kwargs
    )
    builder.pop('relax')  # comment this to also relax

    _, node = run.get_node(builder)
    assert node.is_finished_ok, f'{node} failed: [{node.exit_status}] {node.exit_message}'

    u_sc = node.outputs.hubbard_structure.hubbard.parameters[0].value
    u_ref = 8.1  # eV
    assert abs(u_sc - u_ref) < 0.5


if __name__ == '__main__':
    test_self_consistent_hubbard()
