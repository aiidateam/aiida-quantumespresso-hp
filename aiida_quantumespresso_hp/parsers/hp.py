# -*- coding: utf-8 -*-
import glob, os, re, numpy
from aiida.common.exceptions import InvalidOperation
from aiida.common.datastructures import calc_states
from aiida.orm.data.array import ArrayData
from aiida.orm.data.parameter import ParameterData
from aiida.parsers.parser import Parser
from aiida.parsers.exceptions import OutputParsingError
from aiida_quantumespresso.parsers import QEOutputParsingError
from aiida_quantumespresso_hp.calculations.hp import HpCalculation

class HpParser(Parser):
    """
    Parser implementation for Quantum ESPRESSO Hp calculations 
    """
    _parser_version = '0.1'
    _parser_name = 'AiiDA Quantum ESPRESSO USCF parser'

    def __init__(self, calculation):
        """
        Initialize the instance of HpParser
        """
        if not isinstance(calculation, HpCalculation):
            raise QEOutputParsingError("input calculation must be a HpCalculation")

        self.calculation = calculation

        super(HpParser, self).__init__(calculation)

    def get_linkname_outparams(self):
        """
        Returns the name of the link to the standard output ParameterData
        """
        return 'parameters'

    def get_linkname_hubbard(self):
        """
        Returns the name of the link to the Hubbard output ParameterData
        """
        return 'hubbard'

    def get_linkname_matrices(self):
        """
        Returns the name of the link to the output matrices ArrayData
        """
        return 'matrices'

    def get_linkname_chi(self):
        """
        Returns the name of the link to the output chi ArrayData
        """
        return 'chi'

    def parse_result_template(self):
        """
        Returns a dictionary 
        """
        return {
            'parser_info': '{} v{}'.format(self._parser_name, self._parser_version),
            'parser_warnings': [],
            'warnings': []
        }
        
    def parse_with_retrieved(self, retrieved):
        """
        Parse the results of retrieved nodes

        :param retrieved: dictionary of retrieved nodes
        """
        is_success = True
        output_nodes = []

        # Only allow parsing if calculation is in PARSING state
        state = self.calculation.get_state()
        if state != calc_states.PARSING:
            raise InvalidOperation("calculation not in '{}' state".format(calc_states.PARSING))

        try:
            output_folder = retrieved[self.calculation._get_linkname_retrieved()]
        except KeyError:
            self.logger.error("no retrieved folder found")
            return False, ()

        # Verify the standard output file is present, parse it and attach as output parameters
        try:
            filepath_stdout = output_folder.get_abs_path(self.calculation._OUTPUT_FILE_NAME)
        except OSError as exception:
            self.logger.error("expected output file '{}' was not found".format(filepath))
            return False, ()

        is_success, dict_stdout = self.parse_stdout(filepath_stdout)
        output_nodes.append((self.get_linkname_outparams(), ParameterData(dict=dict_stdout)))

        # The final chi and hubbard files are only written by a serial or post-processing calculation
        complete_calculation = True

        # We cannot use get_abs_path of the output_folder, since that will check for file existence and will throw
        output_path = output_folder.get_abs_path('.')
        filepath_chi = os.path.join(output_path, self.calculation._PREFIX + self.calculation._OUTPUT_CHI_SUFFIX)
        filepath_hubbard = os.path.join(output_path, self.calculation._PREFIX + self.calculation._OUTPUT_HUBBARD_SUFFIX)

        for filepath in [filepath_chi, filepath_hubbard]:
            if not os.path.isfile(filepath):
                complete_calculation = False
                self.logger.info("output file '{}' was not found, assuming partial calculation".format(filepath))

        if complete_calculation:
            dict_hubbard = self.parse_hubbard(filepath_hubbard)
            dict_chi = self.parse_chi(filepath_chi)

            output_matrices = ArrayData()
            output_matrices.set_array('chi0', dict_hubbard['chi0'])
            output_matrices.set_array('chi1', dict_hubbard['chi1'])
            output_matrices.set_array('chi0_inv', dict_hubbard['chi0_inv'])
            output_matrices.set_array('chi1_inv', dict_hubbard['chi1_inv'])
            output_matrices.set_array('hubbard', dict_hubbard['hubbard'])

            output_chi = ArrayData()
            output_chi.set_array('chi0', dict_chi['chi0'])
            output_chi.set_array('chi1', dict_chi['chi1'])

            output_hubbard = ParameterData(dict=dict_hubbard['hubbard_U'])

            output_nodes.append((self.get_linkname_matrices(), output_matrices))
            output_nodes.append((self.get_linkname_hubbard(), output_hubbard))
            output_nodes.append((self.get_linkname_chi(), output_chi))
        
        return is_success, output_nodes

    def parse_stdout(self, filepath):
        """
        Parse the output parameters from the output of a Hp calculation
        written to standard out

        :param filepath: path to file containing output written to stdout
        :returns: boolean representing success status of parsing, True equals parsing was successful
        :returns: dictionary with the parsed parameters
        """
        is_successful = True
        is_terminated = True
        result = self.parse_result_template()

        try:
            with open(filepath, 'r') as handle:
                output = handle.readlines()
        except IOError:
            raise QEOutputParsingError("failed to read file: {}.".format(filepath))

        # Empty output can be considered as a problem
        if not output:
            result['parser_warnings'].append('no_output')
            is_successful = False
            return is_successful, result

        # Parse the output line by line by creating an iterator of the lines
        it = iter(output)
        for line in it:

            # If the output does not contain the line with 'JOB DONE' the program was prematurely terminated
            if 'JOB DONE' in line:
                is_terminated = False

            # If the run did not convergence we expect to find the following string
            match = re.search('.*Convergence has not been reached after\s+([0-9]+)\s+iterations!.*', line)
            if match:
                result['parser_warnings'].append('not_converged')
                is_successful = False

            # Determine the atomic sites that will be perturbed, or that the calculation expects
            # to have been calculated when post-processing the final matrices
            match = re.search('.*List of\s+([0-9]+)\s+atoms which will be perturbed.*', line)
            if match:
                hubbard_sites = {}
                number_of_perturbed_atoms = int(match.group(1))
                blank_line = next(it)
                for i in range(number_of_perturbed_atoms):
                    values = next(it).split()
                    index = values[0]
                    kind = values[1]
                    hubbard_sites[index] = kind
                result['hubbard_sites'] = hubbard_sites

            # A calculation that will only perturb a single atom will only print one line
            match = re.search('.*Atom which will be perturbed.*', line)
            if match:
                hubbard_sites = {}
                number_of_perturbed_atoms = 1
                blank_line = next(it)
                for i in range(number_of_perturbed_atoms):
                    values = next(it).split()
                    index = values[0]
                    kind = values[1]
                    hubbard_sites[index] = kind
                result['hubbard_sites'] = hubbard_sites

        if is_terminated:
            result['parser_warnings'].append('terminated')
            is_successful = False

        return is_successful, result

    def parse_chi(self, filepath):
        """
        Parse the contents of the file {prefix}.chi.dat as written by a HpCalculation

        :param filepath: absolute filepath to the chi.dat output file
        :returns: dictionary with parsed contents
        """
        try:
            with open(filepath, 'r') as handle:
                data = handle.readlines()
        except IOError as exception:
            raise OutputParsingError("could not read the '{}' output file".format(os.path.basename(filepath)))

        result = {}
        blocks = {
            'chi0': [None, None],
            'chi1': [None, None],
        }

        for line_number, line in enumerate(data):
            if 'chi0' in line:
                blocks['chi0'][0] = line_number + 1

            if 'chi1' in line:
                blocks['chi0'][1] = line_number
                blocks['chi1'][0] = line_number + 1
                blocks['chi1'][1] = len(data)
                break

        if not all(sum(blocks.values(), [])):
            raise OutputParsingError("could not determine beginning and end of all blocks in '{}'"
                .format(os.path.basename(filepath)))

        for matrix_name in ('chi0', 'chi1'):
            matrix_block = blocks[matrix_name]
            matrix_data = data[matrix_block[0]:matrix_block[1]]
            matrix = numpy.matrix(self.parse_hubbard_matrix(matrix_data))
            result[matrix_name] = matrix

        return result

    def parse_hubbard(self, filepath):
        """
        Parse the contents of the file {prefix}.Hubbard_U.dat as written by a HpCalculation

        :param filepath: absolute filepath to the Hubbard_U.dat output file
        :returns: dictionary with parsed contents
        """
        try:
            with open(filepath, 'r') as handle:
                data = handle.readlines()
        except IOError as exception:
            raise OutputParsingError("could not read the '{}' output file".format(os.path.basename(filepath)))

        result = {
            'hubbard_U': {
                'sites': []
            }
        }
        blocks = {
            'chi0':     [None, None],
            'chi1':     [None, None],
            'chi0_inv': [None, None],
            'chi1_inv': [None, None],
            'hubbard':  [None, None],
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
                            'kind':  subdata[1],
                            'value': subdata[2],
                        })
                    else:
                        parsed = True

            if 'chi0 matrix' in line:
                blocks['chi0'][0] = line_number + 1

            if 'chi1 matrix' in line:
                blocks['chi0'][1] = line_number
                blocks['chi1'][0] = line_number + 1

            if 'chi0^{-1} matrix' in line:
                blocks['chi1'][1] = line_number
                blocks['chi0_inv'][0] = line_number + 1

            if 'chi1^{-1} matrix' in line:
                blocks['chi0_inv'][1] = line_number
                blocks['chi1_inv'][0] = line_number + 1

            if 'U matrix' in line:
                blocks['chi1_inv'][1] = line_number
                blocks['hubbard'][0] = line_number + 1
                blocks['hubbard'][1] = len(data)
                break

        if not all(sum(blocks.values(), [])):
            raise OutputParsingError("could not determine beginning and end of all matrix blocks in '{}'"
                .format(os.path.basename(filepath)))

        for matrix_name in ('chi0', 'chi1', 'chi0_inv', 'chi1_inv', 'hubbard'):
            matrix_block = blocks[matrix_name]
            matrix_data = data[matrix_block[0]:matrix_block[1]]
            matrix = self.parse_hubbard_matrix(matrix_data)

            if len(set(matrix.shape)) != 1:
                raise OutputParsingError("the matrix '{}' in '{}'' is not square but has shape {}"
                    .format(matrix_name, os.path.basename(filepath), matrix.shape))

            result[matrix_name] = matrix

        return result

    def parse_hubbard_matrix(self, data):
        """
        Utility function to parse one of the matrices that are written to the {prefix}.Hubbard_U.dat
        file by a HpCalculation. Each matrix should be square of size N, which is given by the product
        of the number of q-points and the number of Hubbard species
        Each matrix row is printed with a maximum number of 8 elements per line and each line is followed
        by an empty line. In the parsing of the data, we will use the empty line to detect the end of
        the current matrix row

        :param data: a list of strings representing lines in the Hubbard_U.dat file of a certain matrix
        :returns: square numpy matrix of floats representing the parsed matrix
        """
        matrix = []
        row = []

        for line in data:
            if line.strip():
                for f in line.split():
                    row.append(float(f))
            else:
                if row:
                    matrix.append(row)
                row = []

        return numpy.matrix(matrix)