#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""Example running a pw.x and hp.x in a squence."""
from aiida.engine import run
from aiida.orm import Dict, KpointsData, StructureData, load_code, load_group
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

# =================================================== #
# !!!!!!!!!!!!!!!!!! CHANGE HERE !!!!!!!!!!!!!!!!!!!! #
# =================================================== #
# Load the code configured for ``pw.x`` and ``hp.x``.
# Make sure to replace this string with the label of a
# ``Code`` that you configured in your profile.
hp_code = load_code('pw@localhost')
pw_code = load_code('pw@localhost')

# ===================== RUN PW ======================= #
pw_builder = pw_code.get_builder()

# Create a LiCoO3 crystal structure
a, b, c, d = 1.40803, 0.81293, 4.68453, 1.62585
cell = [[a, -b, c], [0.0, d, c], [-a, -b, c]]
positions = [[0, 0, 0], [0, 0, 3.6608], [0, 0, 10.392], [0, 0, 7.0268]]
symbols = ['Co', 'O', 'O', 'Li']
structure = StructureData(cell=cell)
for position, symbol in zip(positions, symbols):
    structure.append_atom(position=position, symbols=symbol)

# Create a structure data with Hubbard parameters
hubbard_structure = HubbardStructureData.from_structure(structure)
hubbard_structure.initialize_onsites_hubbard('Co', '3d')  # initialize Hubbard atom
hubbard_structure.initialize_intersites_hubbard('Co', '3d', 'O', '2p')  # initialize Hubbard atom
pw_builder.structure = hubbard_structure

# Load the pseudopotential family.
pseudo_family = load_group('SSSP/1.2/PBEsol/efficiency')
pw_builder.pseudos = pseudo_family.get_pseudos(structure=structure)

# Request the recommended wavefunction and charge density cutoffs
# for the given structure and energy units.
cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structure, unit='Ry')

parameters = Dict({
    'CONTROL': {
        'calculation': 'scf'
    },
    'SYSTEM': {
        'ecutwfc': cutoff_wfc,
        'ecutrho': cutoff_rho,
    }
})
pw_builder.parameters = parameters

# Generate a 2x2x2 Monkhorst-Pack mesh
kpoints = KpointsData()
kpoints.set_kpoints_mesh([2, 2, 2])
pw_builder.kpoints = kpoints

# Run the calculation on 1 CPU and kill it if it runs longer than 1800 seconds.
# Set ``withmpi`` to ``False`` if ``pw.x`` was compiled without MPI support.
pw_builder.metadata.options = {
    'resources': {
        'num_machines': 1,
    },
    'max_wallclock_seconds': 1800,
    'withmpi': False,
}

results, pw_node = run.get_node(pw_builder)
print(f'Calculation: {pw_node.process_class}<{pw_node.pk}> {pw_node.process_state.value} [{pw_node.exit_status}]')
print(f'Results: {results}')
assert pw_node.is_finished_ok, f'{pw_node} failed: [{pw_node.exit_status}] {pw_node.exit_message}'

# ===================== RUN HP ======================= #
hp_builder = hp_code.get_builder()

# Assign the remote folder where to take from the
# wavefunctions and other data for the ``hp.x`` to run
parent_scf = pw_node.outputs.remote_folder
hp_builder.parent_scf = parent_scf

parameters = Dict({
    'INPUTHP': {
        'conv_thr_chi': 1.0e-3
    },
})
hp_builder.parameters = parameters

# Generate a 1x1x1 Monkhorst-Pack mesh
qpoints = KpointsData()
qpoints.set_kpoints_mesh([1, 1, 1])
hp_builder.qpoints = qpoints

# Run the calculation on 1 CPU and kill it if it runs longer than 1800 seconds.
# Set ``withmpi`` to ``False`` if ``pw.x`` was compiled without MPI support.
hp_builder.metadata.options = {
    'resources': {
        'num_machines': 1,
    },
    'max_wallclock_seconds': 1800,
    'withmpi': False,
}

results, node = run.get_node(hp_builder)
print(f'Calculation: {node.process_class}<{node.pk}> {node.process_state.value} [{node.exit_status}]')
print(f'Results: {results}')
assert node.is_finished_ok, f'{node} failed: [{node.exit_status}] {node.exit_message}'
