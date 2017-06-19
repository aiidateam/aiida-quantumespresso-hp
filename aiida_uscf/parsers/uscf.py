# -*- coding: utf-8 -*-

import numpy as np
from aiida.common.exceptions import InvalidOperation
from aiida.common.datastructures import calc_states
from aiida.parsers.parser import Parser
from aiida.orm import CalculationFactory
from aiida.orm.data.array import ArrayData
from aiida.orm.data.parameter import ParameterData
from aiida.parsers.plugins.quantumespresso import QEOutputParsingError

UscfCalculation = CalculationFactory('quantumespresso.uscf')

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

        output_matrices = ArrayData()
        output_matrices.set_array('chi0', np.matrix(dict_hubbard['matrix_chi0']))
        output_matrices.set_array('chi1', np.matrix(dict_hubbard['matrix_chi1']))
        output_matrices.set_array('chi0_inv', np.matrix(dict_hubbard['matrix_chi0_inv']))
        output_matrices.set_array('chi1_inv', np.matrix(dict_hubbard['matrix_chi1_inv']))
        output_matrices.set_array('hubbard', np.matrix(dict_hubbard['matrix_hubbard']))

        output_params = ParameterData(dict=dict_stdout)
        output_hubbard = ParameterData(dict=dict_hubbard['hubbard_U'])

        output_nodes = [
            (self.get_linkname_outparams(), output_params),
            (self.get_linkname_hubbard(), output_hubbard),
            (self.get_linkname_matrices(), output_matrices)
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
        result = {
            'chi0': [],
            'chi1': []
        }

        with open(filepath, 'r') as handle:
            data = handle.readlines()

        for line_number, line in enumerate(data):
            if 'chi0' in line:
                offset = line_number + 1
                for i in range(8):
                    result['chi0'].append(float(data[offset + i]))

            if 'chi1' in line:
                offset = line_number + 1
                for i in range(8):
                    result['chi0'].append(float(data[offset + i]))

        return result

    def parse_hubbard(self, filepath):
        """
        Parse the contents of the file {prefix}.Hubbard_U.dat as written by a UscfCalculation

        :param filepath: absolute filepath to the Hubbard_U.dat output file
        :returns: dictionary with parsed contents
        """
        result = {
            'hubbard_U': {}
        }

        with open(filepath, 'r') as handle:
            data = handle.readlines()

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
                offset = line_number + 1
                result['matrix_chi0'] = self.parse_hubbard_matrix(data, offset)

            if 'chi1 matrix' in line:
                offset = line_number + 1
                result['matrix_chi1'] = self.parse_hubbard_matrix(data, offset)

            if 'chi0^{-1} matrix' in line:
                offset = line_number + 1
                result['matrix_chi0_inv'] = self.parse_hubbard_matrix(data, offset)

            if 'chi1^{-1} matrix' in line:
                offset = line_number + 1
                result['matrix_chi1_inv'] = self.parse_hubbard_matrix(data, offset)

            if 'U matrix' in line:
                offset = line_number + 1
                result['matrix_hubbard'] = self.parse_hubbard_matrix(data, offset)

        return result

    def parse_hubbard_matrix(self, data, offset):
        """
        Utility function to parse one of the matrices that are written to the {prefix}.Hubbard_U.dat
        file by a UscfCalculation. The matrices typically have size 8x8 and have empty white lines
        in between them.

        :param data: a list of of strings representing each line in the Hubbar_U.dat file
        :param offset: integer indicating the linenumber offset at which to start reading the matrix
        """
        matrix = []

        for i in range(8):
            offset_row = i * 2
            matrix_row = data[offset + offset_row]
            matrix.append([float(f) for f in matrix_row.split()])

        return matrix