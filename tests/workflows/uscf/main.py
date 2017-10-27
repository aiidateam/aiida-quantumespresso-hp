#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

import argparse
from aiida.work.run import run
from aiida.orm import Code, CalculationFactory
from aiida.orm.data.base import Bool, Int
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.common.exceptions import NotExistent
from aiida_quantumespresso_uscf.workflows.uscf.main import UscfWorkChain

PwCalculation = CalculationFactory('quantumespresso.pw')
UscfCalculation = CalculationFactory('quantumespresso.uscf')

def parser_setup():
    """
    Setup the parser of command line arguments and return it. This is separated from the main
    execution body to allow tests to effectively mock the setup of the parser and the command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Perform a calculation of Quantum ESPRESSO Uscf.x for a previously completed PwCalculation',
    )
    parser.add_argument(
        '-a', action="store_true", dest='parallelize_atoms',
        help='parallelize the calculations over individual atomic calculations'
    )
    parser.add_argument(
        '-m', type=int, default=5, dest='max_iterations',
        help='the maximum number of iterations to allow for each SCF cycle for a single k-point. (default: %(default)d)'
    )
    parser.add_argument(
        '-k', nargs=3, type=int, default=[2, 2, 2], dest='qpoints', metavar='Q',
        help='define the q-points mesh. (default: %(default)s)'
    )
    parser.add_argument(
        '-c', type=str, required=True, dest='codename',
        help='the label of the AiiDA code instance that references QE Uscf.x'
    )
    parser.add_argument(
        '-p', type=int, required=True, dest='parent_calculation',
        help='the node id of the parent PwCalculation'
    )
    parser.add_argument(
        '-w', type=int, default=1800, dest='max_wallclock_seconds',
        help='the maximum wallclock time in seconds to set for the calculations. (default: %(default)d)'
    )

    return parser


def execute(args):
    """
    The main execution of the script, which will run some preliminary checks on the command
    line arguments before passing them to the workchain and running it
    """
    try:
        code = Code.get_from_string(args.codename)
    except NotExistent as exception:
        print "Execution failed: could not retrieve the code '{}'".format(args.codename)
        print "Exception report: {}".format(exception)
        return

    try:
        parent_calculation = load_node(args.parent_calculation)
    except NotExistent as exception:
        print "Execution failed: failed to load the node for the given parent calculation '{}'".format(args.parent_calculation)
        print "Exception report: {}".format(exception)
        return

    if not isinstance(parent_calculation, PwCalculation):
        print "The provided parent calculation {} is not of type {}, aborting...".format(args.parent_calculation, 'PwCalculation')
        return

    try:
        qpoints = KpointsData()
        qpoints.set_kpoints_mesh(args.qpoints)
    except Exception as exception:
        print "Execution failed: failed to instantiate a KpointsData object from the specified q-points mesh"
        print "Exception report: {}".format(exception)
        return

    parameters = {
        'INPUTUSCF': {
        }
    }
    settings = {}
    options  = {
        'resources': {
            'num_machines': 1
        },
        'max_wallclock_seconds': args.max_wallclock_seconds,
    }

    run(
        UscfWorkChain,
        code=code,
        parent_calculation=parent_calculation,
        qpoints=qpoints,
        parameters=ParameterData(dict=parameters),
        settings=ParameterData(dict=settings),
        options=ParameterData(dict=options),
        max_iterations=Int(args.max_iterations),
        parallelize_atoms=Bool(args.parallelize_atoms)
    )


def main():
    """
    Setup the parser to retrieve the command line arguments and pass them to the main execution function.
    """
    parser = parser_setup()
    args   = parser.parse_args()
    result = execute(args)


if __name__ == "__main__":
    main()