# -*- coding: utf-8 -*-
"""A collection of function that are used to parse the output of Quantum Espresso HP.

The function that needs to be called from outside is parse_raw_output().
The functions mostly work without aiida specific functionalities.
"""
import re

from aiida_quantumespresso.utils.mapping import get_logging_container


def parse_raw_output(stdout):
    """Parse the output parameters from the output of a Hp calculation written to standard out.

    :param filepath: path to file containing output written to stdout
    :returns: boolean representing success status of parsing, True equals parsing was successful
    :returns: dictionary with the parsed parameters
    """
    parsed_data = {}
    logs = get_logging_container()
    is_prematurely_terminated = True

    # Parse the output line by line by creating an iterator of the lines
    iterator = iter(stdout.split('\n'))
    for line in iterator:

        # If the output does not contain the line with 'JOB DONE' the program was prematurely terminated
        if 'JOB DONE' in line:
            is_prematurely_terminated = False

        if 'reading inputhp namelist' in line:
            logs.error.append('ERROR_INVALID_NAMELIST')

        # If the atoms were not ordered correctly in the parent calculation
        if 'WARNING! All Hubbard atoms must be listed first in the ATOMIC_POSITIONS card of PWscf' in line:
            logs.error.append('ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS')

        # If the calculation run out of walltime we expect to find the following string
        match = re.search(r'.*Maximum CPU time exceeded.*', line)
        if match:
            logs.error.append('ERROR_OUT_OF_WALLTIME')

        # If not all expected perturbation files were found for a chi_collect calculation
        if 'Error in routine hub_read_chi (1)' in line:
            logs.error.append('ERROR_MISSING_PERTURBATION_FILE')

        # If the run did not convergence we expect to find the following string
        match = re.search(r'.*Convergence has not been reached after\s+([0-9]+)\s+iterations!.*', line)
        if match:
            logs.error.append('ERROR_CONVERGENCE_NOT_REACHED')

        # A calculation that will only perturb a single atom will only print one line
        match = re.search(r'.*The grid of q-points.*\s+([0-9])+\s+q-points.*', line)
        if match:
            ### DEBUG
            print(int(match.group(1)))
            ### DEBUG
            parsed_data['number_of_qpoints'] = int(match.group(1))

        # Determine the atomic sites that will be perturbed, or that the calculation expects
        # to have been calculated when post-processing the final matrices
        match = re.search(r'.*List of\s+([0-9]+)\s+atoms which will be perturbed.*', line)
        if match:
            hubbard_sites = {}
            number_of_perturbed_atoms = int(match.group(1))
            _ = next(iterator)  # skip blank line
            for _ in range(number_of_perturbed_atoms):
                values = next(iterator).split()
                index = values[0]
                kind = values[1]
                hubbard_sites[index] = kind
            parsed_data['hubbard_sites'] = hubbard_sites

        # A calculation that will only perturb a single atom will only print one line
        match = re.search(r'.*Atom which will be perturbed.*', line)
        if match:
            hubbard_sites = {}
            number_of_perturbed_atoms = 1
            _ = next(iterator)  # skip blank line
            for _ in range(number_of_perturbed_atoms):
                values = next(iterator).split()
                index = values[0]
                kind = values[1]
                hubbard_sites[index] = kind
            parsed_data['hubbard_sites'] = hubbard_sites

    if is_prematurely_terminated:
        logs.error.append('ERROR_OUTPUT_STDOUT_INCOMPLETE')

    return parsed_data, logs
