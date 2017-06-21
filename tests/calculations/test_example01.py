#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

import os
import argparse
import pymatgen
from aiida.work.run import run
from aiida.orm.data.upf import UpfData
from aiida.common.exceptions import NotExistent
from aiida.orm.data.upf import get_pseudos_from_structure
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from seekpath.aiidawrappers import get_path
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso_uscf.calculations.uscf import UscfCalculation


def parser_setup():
    """
    Setup the parser of command line arguments and return it. This is separated from the main
    execution body to allow tests to effectively mock the setup of the parser and the command line arguments
    """
    parser = argparse.ArgumentParser(
        description="""Example calculation of the Uscf module for Quantum ESPRESSO.
        Computes Hubbard U self-consistenly for a simple LiCoO2 system""",
    )
    parser.add_argument(
        '-p', type=str, required=True, dest='pseudo_family',
        help='the name of pseudo family to use'
    )
    parser.add_argument(
        '-u', type=str, required=True, dest='codename_uscf',
        help='the label of the AiiDA code that references Uscf.x'
    )
    parser.add_argument(
        '-w', type=str, required=True, dest='codename_pw',
        help='the label of the AiiDA code that references QE pw.x'
    )
    parser.add_argument(
        '-c', type=str, dest='parent_calculation',
        help='the pk of the parent PwCalculation if it was already completed and you want to skip it'
    )
    parser.add_argument(
        '-t', type=int, default=1800, dest='max_wallclock_seconds',
        help='the maximum wallclock time in seconds to set for the calculations. (default: %(default)d)'
    )

    return parser


def execute(args):
    """
    The main execution of the script, which will run some preliminary checks on the command
    line arguments before passing them to the workchain and running it
    """
    structure = construct_structure()

    options = {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
        },
        'max_wallclock_seconds': args.max_wallclock_seconds,
    }

    try:
        code_pw = Code.get_from_string(args.codename_pw)
    except NotExistent as exception:
        print "Execution failed: could not retrieve the code '{}'".format(args.codename_pw)
        print "Exception report: {}".format(exception)
        return

    try:
        code_uscf = Code.get_from_string(args.codename_uscf)
    except NotExistent as exception:
        print "Execution failed: could not retrieve the code '{}'".format(args.codename_uscf)
        print "Exception report: {}".format(exception)
        return

    try:
        pseudo_family = UpfData.get_upf_group(args.pseudo_family)
        pseudo = get_pseudos_from_structure(structure, args.pseudo_family)
    except NotExistent as exception:
        print "Execution failed: could not retrieve the pseudo family group '{}'".format(args.pseudo_family)
        print "Exception report: {}".format(exception)
        return

    if args.parent_calculation:
        try:
            parent = load_node(int(args.parent_calculation))
        except NotExistent as exception:
            print "Execution failed: could not retrieve the specified parent calculation '{}'".format(args.parent_calculation)
            print "Exception report: {}".format(exception)
            return
        print "Successfully loaded the parent calculation {}<{}>".format(type(parent), parent.pk)
    else:
        result, pk = run_pw(code_pw, structure, pseudo, options)
        parent = load_node(pk)
        print "Successfully completed the parent calculation {}<{}>".format(type(parent), parent.pk)

    result, pk = run_uscf(code_uscf, parent, options)
    
    try:
        output_hubbard = result['output_hubbard']
        print "UscfCalculation finished with the following Hubbard U parameters:"
        print output_hubbard.get_dict()
    except KeyError as exception:
        print "UscfCalculation did not return an 'output_hubbard' node, it probably failed"


def run_pw(code_pw, structure, pseudo, options):
    """
    Run the self-consistent field calculation with QE's pw.x
    """
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2])

    parameters = {
        'CONTROL': {
            'restart_mode': 'from_scratch',
            'calculation': 'scf',
        },
        'SYSTEM': {
            'nosym': True,
            'noinv': True,
            'ecutwfc': 30.,
            'ecutrho': 240.,
            'lda_plus_u': True,
            'lda_plus_u_kind': 0,
            'hubbard_u': {'Co': 1.E-8},
        },
        'ELECTRONS': {
            'conv_thr': 1.E-10,
            'mixing_beta': 0.7
        }
    }

    inputs = {
        'code': code_pw,
        'pseudo': pseudo,
        'kpoints': kpoints,
        'structure': structure,
        'parameters': ParameterData(dict=parameters),
        '_options': options,
    }

    process = PwCalculation.process()
    result, pk = run(process, _return_pid=True, **inputs)

    return result, pk


def run_uscf(code_uscf, parent_calculation, options):
    """
    Run the self-consistent Hubbard calculation with QE's Uscf.x
    """
    qpoints = KpointsData()
    qpoints.set_kpoints_mesh([2, 2, 2])

    parameters = {
        'INPUTUSCF': {
        }
    }

    inputs = {
        'code': code_uscf,
        'qpoints': qpoints,
        'parameters': ParameterData(dict=parameters),
        '_options': options,
        'parent_folder': parent_calculation.out.remote_folder,
    }

    process = UscfCalculation.process()
    result, pk = run(process, _return_pid=True, **inputs)

    return result, pk


def construct_structure():
    """
    LiCoO2 (COD #4505482)
    """
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '../resources/cif/LiCoO2.cif')
    struct_pm = pymatgen.Structure.from_file(filepath)
    structure = StructureData(pymatgen=struct_pm)
    rseekpath = get_path(structure)
    primitive = rseekpath.pop('primitive_structure')

    return primitive


def main():
    """
    Setup the parser to retrieve the command line arguments and pass them to the main execution function.
    """
    parser = parser_setup()
    args   = parser.parse_args()
    result = execute(args)


if __name__ == "__main__":
    main()