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

        detect_important_message(logs, line)

        # A calculation that will only perturb a single atom will only print one line
        match = re.search(r'.*The grid of q-points.*?(\d+)+\s+q-points.*', line)
        if match:
            parsed_data['number_of_qpoints'] = int(match.group(1))

        # Determine the atomic sites that will be perturbed, or that the calculation expects
        # to have been calculated when post-processing the final matrices
        match = re.search(r'.*List of.*?(\d+)\s+atoms which will be perturbed.*', line)
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

    # Remove duplicate log messages by turning it into a set. Then convert back to list as that is what is expected
    logs.error = list(set(logs.error))
    logs.warning = list(set(logs.warning))

    return parsed_data, logs


def detect_important_message(logs, line):
    """Detect error or warning messages, and append to the log if a match is found."""
    REG_ERROR_CONVERGENCE_NOT_REACHED = re.compile(
        r'.*Convergence has not been reached after\s+([0-9]+)\s+iterations!.*'
    )
    ERROR_POSITIONS = 'WARNING! All Hubbard atoms must be listed first in the ATOMIC_POSITIONS card of PWscf'
    message_map = {
        'error': {
            'Error in routine hub_read_chi (1)': 'ERROR_MISSING_PERTURBATION_FILE',
            'Maximum CPU time exceeded': 'ERROR_OUT_OF_WALLTIME',
            'reading inputhp namelist': 'ERROR_INVALID_NAMELIST',
            'problems computing cholesky': 'ERROR_COMPUTING_CHOLESKY',
            'Reconstruction problem: some chi were not found': 'ERROR_MISSING_CHI_MATRICES',
            'incompatible FFT grid': 'ERROR_INCOMPATIBLE_FFT_GRID',
            REG_ERROR_CONVERGENCE_NOT_REACHED: 'ERROR_CONVERGENCE_NOT_REACHED',
            ERROR_POSITIONS: 'ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS',
            'WARNING: The Fermi energy shift is zero or too big!': 'ERROR_FERMI_SHIFT',
        },
        'warning': {
            'Warning:': None,
            'DEPRECATED:': None,
        }
    }

    # Match any known error and warning messages
    for marker, message in message_map['error'].items():
        # Replace with isinstance(marker, re.Pattern) once Python 3.6 is dropped
        if hasattr(marker, 'search'):
            if marker.match(line):
                if message is None:
                    message = line
                logs.error.append(message)
        else:
            if marker in line:
                if message is None:
                    message = line
                logs.error.append(message)

    for marker, message in message_map['warning'].items():
        if marker in line:
            if message is None:
                message = line
            logs.warning.append(message)
