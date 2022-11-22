# -*- coding: utf-8 -*-
"""Parser implementation for the `HpCalculation` plugin."""
import os
import re

from aiida import orm
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida_quantumespresso.utils.mapping import get_logging_container
import numpy

from aiida_quantumespresso_hp.calculations.hp import HpCalculation


class HpParser(Parser):
    """Parser implementation for the `HpCalculation` plugin."""

    def parse(self, **kwargs):
        """Parse the contents of the output files retrieved in the `FolderData`."""
        try:
            self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        for parse_method in [
            self.parse_stdout, self.parse_hubbard, self.parse_hubbard_chi, self.parse_hubbard_parameters
        ]:
            exit_code = parse_method()
            if exit_code:
                return exit_code

    @property
    def is_initialization_only(self):
        """Return whether the calculation was an `initialization_only` run.

        This is the case if the `determin_num_pert_only` flag was set to `True` in the `INPUTHP` namelist.
        In this case, there will only be a stdout file. All other output files will be missing, but that is expected.
        """
        return self.node.inputs.parameters.base.attributes.get('INPUTHP', {}).get('determine_num_pert_only', False)

    @property
    def is_partial_site(self):
        """Return whether the calculation computed just a sub set of all sites to be perturbed.

        A complete run means that all perturbations were calculation and the final matrices were computerd
        """
        card = self.node.inputs.parameters.base.attributes.get('INPUTHP', {})
        return any(key.startswith('perturb_only_atom') for key in card.keys())

    @property
    def is_complete_calculation(self):
        """Return whether the calculation was a complete run.

        A complete run means that all perturbations were performed and the final matrices were computed.
        """
        return not (self.is_initialization_only or self.is_partial_site)

    def parse_stdout(self):
        """Parse the stdout output file.

        Parse the output parameters from the output of a Hp calculation written to standard out.

        :return: optional exit code in case of an error
        """
        filename = self.node.base.attributes.get('output_filename')

        if filename not in self.retrieved.base.repository.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            stdout = self.retrieved.base.repository.get_object_content(filename)
        except IOError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        try:
            parsed_data, logs = self.parse_stdout_content(stdout)
        except Exception:  # pylint: disable=broad-except
            return self.exit_codes.ERROR_OUTPUT_STDOUT_PARSE
        else:
            self.out('parameters', orm.Dict(dict=parsed_data))

        exit_statuses = [
            'ERROR_INVALID_NAMELIST',
            'ERROR_OUTPUT_STDOUT_INCOMPLETE',
            'ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS',
            'ERROR_MISSING_PERTURBATION_FILE',
            'ERROR_CONVERGENCE_NOT_REACHED',
        ]

        for exit_status in exit_statuses:
            if exit_status in logs['error']:
                return self.exit_codes.get(exit_status)

    def parse_hubbard(self):
        """Parse the hubbard output file.

        :return: optional exit code in case of an error
        """
        filename = HpCalculation.filename_output_hubbard

        try:
            with self.retrieved.base.repository.open(filename, 'r') as handle:
                parsed_data = self.parse_hubbard_content(handle)
        except IOError:
            if self.is_complete_calculation:
                return self.exit_codes.ERROR_OUTPUT_HUBBARD_MISSING
        else:
            matrices = orm.ArrayData()
            matrices.set_array('chi', parsed_data['chi'])
            matrices.set_array('chi0', parsed_data['chi0'])
            matrices.set_array('chi_inv', parsed_data['chi_inv'])
            matrices.set_array('chi0_inv', parsed_data['chi0_inv'])
            matrices.set_array('hubbard', parsed_data['hubbard'])

            self.out('hubbard', orm.Dict(dict=parsed_data['hubbard_U']))
            self.out('hubbard_matrices', matrices)

    def parse_hubbard_chi(self):
        """Parse the hubbard chi output file.

        :return: optional exit code in case of an error
        """
        filename = HpCalculation.filename_output_hubbard_chi

        try:
            with self.retrieved.base.repository.open(filename, 'r') as handle:
                parsed_data = self.parse_chi_content(handle)
        except IOError:
            if self.is_complete_calculation:
                return self.exit_codes.ERROR_OUTPUT_HUBBARD_CHI_MISSING
        else:
            output_chi = orm.ArrayData()
            output_chi.set_array('chi', parsed_data['chi'])
            output_chi.set_array('chi0', parsed_data['chi0'])

            self.out('hubbard_chi', output_chi)

    def parse_hubbard_parameters(self):
        """Parse the hubbard parameters output file.

        :return: optional exit code in case of an error
        """
        filename = HpCalculation.filename_output_hubbard_parameters

        try:
            with self.retrieved.base.repository.open(filename, 'rb') as handle:
                self.out('hubbard_parameters', orm.SinglefileData(file=handle))
        except IOError:
            pass  # the file is not required to exist

    @staticmethod
    def parse_stdout_content(stdout):
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

            # If not all expected perturbation files were found for a chi_collect calculation
            if 'Error in routine hub_read_chi (1)' in line:
                logs.error.append('ERROR_MISSING_PERTURBATION_FILE')

            # If the run did not convergence we expect to find the following string
            match = re.search(r'.*Convergence has not been reached after\s+([0-9]+)\s+iterations!.*', line)
            if match:
                logs.error.append('ERROR_CONVERGENCE_NOT_REACHED')

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

    def parse_chi_content(self, handle):
        """Parse the contents of the file {prefix}.chi.dat as written by a HpCalculation.

        :param filepath: absolute filepath to the chi.dat output file
        :returns: dictionary with parsed contents
        """
        data = handle.readlines()

        result = {}
        blocks = {
            'chi': [None, None],
            'chi0': [None, None],
        }

        for line_number, line in enumerate(data):
            if 'chi0 :' in line:
                blocks['chi0'][0] = line_number + 1

            if 'chi :' in line:
                blocks['chi0'][1] = line_number
                blocks['chi'][0] = line_number + 1
                blocks['chi'][1] = len(data)
                break

        if not all(sum(list(blocks.values()), [])):
            raise ValueError(
                f"could not determine beginning and end of all blocks in '{os.path.basename(handle.name)}'"
            )

        for matrix_name in ('chi0', 'chi'):
            matrix_block = blocks[matrix_name]
            matrix_data = data[matrix_block[0]:matrix_block[1]]
            matrix = numpy.array(self.parse_hubbard_matrix(matrix_data))
            result[matrix_name] = matrix

        return result

    def parse_hubbard_content(self, handle):
        """Parse the contents of the file {prefix}.Hubbard_U.dat as written by a HpCalculation.

        :param filepath: absolute filepath to the Hubbard_U.dat output file
        :returns: dictionary with parsed contents
        """
        data = handle.readlines()

        result = {'hubbard_U': {'sites': []}}
        blocks = {
            'chi': [None, None],
            'chi0': [None, None],
            'chi_inv': [None, None],
            'chi0_inv': [None, None],
            'hubbard': [None, None],
        }

        for line_number, line in enumerate(data):

            if 'site n.' in line:
                parsed = False
                subline_number = line_number + 1
                while not parsed:
                    subline = data[subline_number].strip()
                    if subline:
                        subline_number += 1
                        subdata = subline.split()
                        result['hubbard_U']['sites'].append({
                            'index': subdata[0],
                            'type': subdata[1],
                            'kind': subdata[2],
                            'spin': subdata[3],
                            'new_type': subdata[4],
                            'new_kind': subdata[5],
                            'value': subdata[6],
                        })
                    else:
                        parsed = True

            if 'chi0 matrix' in line:
                blocks['chi0'][0] = line_number + 1

            if 'chi matrix' in line:
                blocks['chi0'][1] = line_number
                blocks['chi'][0] = line_number + 1

            if 'chi0^{-1} matrix' in line:
                blocks['chi'][1] = line_number
                blocks['chi0_inv'][0] = line_number + 1

            if 'chi^{-1} matrix' in line:
                blocks['chi0_inv'][1] = line_number
                blocks['chi_inv'][0] = line_number + 1

            if 'Hubbard matrix' in line:
                blocks['chi_inv'][1] = line_number
                blocks['hubbard'][0] = line_number + 1
                blocks['hubbard'][1] = len(data)
                break

        if not all(sum(list(blocks.values()), [])):
            raise ValueError(
                f'could not determine beginning and end of all matrix blocks in `{os.path.basename(handle.name)}`'
            )

        for matrix_name in ('chi0', 'chi', 'chi0_inv', 'chi_inv', 'hubbard'):
            matrix_block = blocks[matrix_name]
            matrix_data = data[matrix_block[0]:matrix_block[1]]
            matrix = self.parse_hubbard_matrix(matrix_data)

            if len(set(matrix.shape)) != 1:
                filename = os.path.basename(handle.name)
                raise ValueError(
                    f'matrix `{matrix_name}` in `{filename}` is not square but has shape {matrix.shape}: {matrix}'
                )

            result[matrix_name] = matrix

        return result

    @staticmethod
    def parse_hubbard_matrix(data):
        """Parse one of the matrices that are written to the {prefix}.Hubbard_U.dat files.

        Each matrix should be square of size N, which is given by the product of the number of q-points and the number
        of Hubbard species. Each matrix row is printed with a maximum number of 8 elements per line and each line is
        followed by an empty line. In the parsing of the data, we will use the empty line to detect the end of the
        current matrix row.

        :param data: a list of strings representing lines in the Hubbard_U.dat file of a certain matrix
        :returns: square numpy matrix of floats representing the parsed matrix
        """
        matrix = []
        row = []

        for line in data:
            if line.strip():
                for value in line.split():
                    row.append(float(value))
            else:
                if row:
                    matrix.append(row)
                row = []

        if row:
            matrix.append(row)

        return numpy.array(matrix)
