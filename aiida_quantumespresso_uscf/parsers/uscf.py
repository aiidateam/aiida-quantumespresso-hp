# -*- coding: utf-8 -*-

import os
import numpy as np
from collections import namedtuple
from aiida.common.exceptions import InvalidOperation
from aiida.common.datastructures import calc_states
from aiida.parsers.exceptions import OutputParsingError
from aiida.orm import CalculationFactory
from aiida.orm.data.array import ArrayData
from aiida.orm.data.parameter import ParameterData
from aiida.parsers.parser import Parser
from aiida_quantumespresso.parsers import QEOutputParsingError
from aiida_quantumespresso_uscf.calculations.uscf import UscfCalculation

class UscfParser(Parser):
    """
    Parser implementation for Uscf (HUBBARD) calculations for Quantum ESPRESSO
    """

    def __init__(self, calculation):
        """
        Initialize the instance of PhParser
        """
        if not isinstance(calculation, UscfCalculation):
            raise QEOutputParsingError("input calculation must be a UscfCalculation")

        self._calc = calculation
        self._OUTPUT_MATRIX_CHI0 = 'output_matrix_chi0'
        self._OUTPUT_MATRIX_CHI1 = 'output_matrix_chi1'
        self._OUTPUT_MATRIX_CHI0_INV = 'output_matrix_chi0_inv'
        self._OUTPUT_MATRIX_CHI1_INV = 'output_matrix_chi1_inv'
        self._OUTPUT_MATRIX_HUBBARD = 'output_matrix_hubbard'

        super(UscfParser, self).__init__(calculation)

    def get_linkname_hubbard(self):
        """
        Returns the name of the link to the Hubbard output ParameterData
        """
        return 'output_hubbard'

    def get_linkname_matrices(self):
        """
        Returns the name of the link to the output matrices ArrayData
        """
        return 'output_matrices'

    def get_linkname_chi(self):
        """
        Returns the name of the link to the output chi ArrayData
        """
        return 'output_chi'
        
    def parse_with_retrieved(self, retrieved):
        """
        Parse the results of retrieved nodes

        :param retrieved: dictionary of retrieved nodes
        """
        is_success = True

        # Only allow parsing if calculation is in PARSING state
        state = self._calc.get_state()
        if state != calc_states.PARSING:
            raise InvalidOperation("calculation not in '{}' state".format(calc_states.PARSING))

        try:
            output_folder = retrieved[self._calc._get_linkname_retrieved()]
        except KeyError:
            self.logger.error("no retrieved folder found")
            return False, ()

        filepath_stdout = output_folder.get_abs_path(self._calc._OUTPUT_FILE_NAME)
        filepath_chi = output_folder.get_abs_path(self._calc._PREFIX + self._calc._OUTPUT_CHI_SUFFIX)
        filepath_hubbard = output_folder.get_abs_path(self._calc._PREFIX + self._calc._OUTPUT_HUBBARD_SUFFIX)

        for filepath in [filepath_stdout, filepath_chi, filepath_hubbard]:
            if not filepath in [output_folder.get_abs_path(f) for f in output_folder.get_folder_list()]:
                self.logger.error("expected output file '{}' was not found".format(filepath))
                return False, ()

        result_stdout, dict_stdout = self.parse_stdout(filepath_stdout)
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

        output_params = ParameterData(dict=dict_stdout)
        output_hubbard = ParameterData(dict=dict_hubbard['hubbard_U'])

        output_nodes = [
            (self.get_linkname_outparams(), output_params),
            (self.get_linkname_matrices(), output_matrices),
            (self.get_linkname_hubbard(), output_hubbard),
            (self.get_linkname_chi(), output_chi)
        ]
        
        return is_success, output_nodes

    def parse_stdout(self, filepath):
        """
        Parse the output parameters from the output of a Uscf calculation
        written to standard out

        :param filepath: path to file containing output written to stdout
        :returns: boolean representing success status of parsing, True equals parsing was successful
        :returns: dictionary with the parsed parameters
        """
        is_success = True
        is_finished_run = False

        parser_version = '0.1'
        parser_info = {}
        parser_info['parser_warnings'] = []
        parser_info['parser_info'] = 'AiiDA QE-USCF parser v{}'.format(parser_version)

        try:
            with open(filepath, 'r') as handle:
                output = handle.readlines()
        except IOError:
            raise QEOutputParsingError("failed to read file: {}.".format(filepath))

        if not output:
            is_success = False

        # Check if the job reached the end (does not guarantee successful execution)
        for line in output[::-1]:
            if 'JOB DONE' in line:
                is_finished_run = True
                break

        if not is_finished_run:
            warning = 'the Uscf calculation did not reach the end of execution'
            parser_info['parser_warnings'].append(warning)        
            is_success = False

        return is_success, {'test_result': 5}

    def parse_chi(self, filepath):
        """
        Parse the contents of the file {prefix}.chi.dat as written by a UscfCalculation

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
            matrix = np.matrix(self.parse_hubbard_matrix(matrix_data))
            result[matrix_name] = matrix

        return result

    def parse_hubbard(self, filepath):
        """
        Parse the contents of the file {prefix}.Hubbard_U.dat as written by a UscfCalculation

        :param filepath: absolute filepath to the Hubbard_U.dat output file
        :returns: dictionary with parsed contents
        """
        try:
            with open(filepath, 'r') as handle:
                data = handle.readlines()
        except IOError as exception:
            raise OutputParsingError("could not read the '{}' output file".format(os.path.basename(filepath)))

        result = {'hubbard_U': {}}
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
                        key = 'Hubbard_U({})'.format(subdata[0])
                        result['hubbard_U'][key] = subdata[2]
                    else:
                        parsed = True

            if 'chi0 matrix' in line:
                blocks['chi0'][0]     = line_number + 1

            if 'chi1 matrix' in line:
                blocks['chi0'][1]     = line_number
                blocks['chi1'][0]     = line_number + 1

            if 'chi0^{-1} matrix' in line:
                blocks['chi1'][1]     = line_number
                blocks['chi0_inv'][0] = line_number + 1

            if 'chi1^{-1} matrix' in line:
                blocks['chi0_inv'][1] = line_number
                blocks['chi1_inv'][0] = line_number + 1

            if 'U matrix' in line:
                blocks['chi1_inv'][1] = line_number
                blocks['hubbard'][0]  = line_number + 1
                blocks['hubbard'][1]  = len(data)
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
        file by a UscfCalculation. Each matrix should be square of size N, which is given by the product
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

        return np.matrix(matrix)