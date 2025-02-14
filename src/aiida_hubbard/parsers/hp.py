# -*- coding: utf-8 -*-
"""Parser implementation for the `HpCalculation` plugin."""
import os

from aiida import orm
from aiida.common import exceptions
from aiida.parsers import Parser
import numpy as np

from aiida_hubbard.calculations.hp import HpCalculation


class HpParser(Parser):
    """Parser implementation for the `HpCalculation` plugin."""

    def parse(self, **kwargs):
        """Parse the contents of the output files retrieved in the `FolderData`."""
        self.exit_code_stdout = None  # pylint: disable=attribute-defined-outside-init

        try:
            self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        # The stdout is always parsed by default.
        logs = self.parse_stdout()

        # Check for specific known problems that can cause a pre-mature termination of the calculation
        exit_code = self.validate_premature_exit(logs)
        if exit_code:
            return exit_code

        if self.exit_code_stdout:
            return self.exit_code_stdout

        # If it only initialized, then we do NOT parse the `{prefix}.Hubbard_parameters.dat``
        # and the {prefix}.chi.dat files.
        # This check is needed since the `hp.x` routine will print the `{prefix}.Hubbard_parameters.dat`
        # also when it is only initialized.
        if not self.is_initialization_only and not self.is_partial_mesh:

            for parse_method in [
                self.parse_hubbard,
                self.parse_hubbard_chi,
            ]:
                exit_code = parse_method()
                if exit_code:
                    return exit_code

        # If the calculation is `complete`, we try to store the parameters in `HubbardStructureData`.
        if self.is_complete_calculation:
            # If the `HUBBARD.dat` is not produced, it means it is an "only Hubbard U" calculation,
            # thus we set the Hubbard parameters from the ``hubbard`` output, which contains the onsite U.
            try:
                self.parse_hubbard_dat(kwargs['retrieved_temporary_folder'])
            except (KeyError, FileNotFoundError):
                self.get_hubbard_structure()

    @property
    def is_initialization_only(self):
        """Return whether the calculation was an `initialization_only` run.

        This is the case if the `determin_num_pert_only` flag was set to `True` in the `INPUTHP` namelist.
        In this case, there will only be a stdout file. All other output files will be missing, but that is expected.
        """
        return self.node.inputs.parameters.base.attributes.get('INPUTHP', {}).get('determine_num_pert_only', False)

    @property
    def is_partial_mesh(self):
        """Return whether the calculation was a run on a qpoint subset.

        This is the case if the `determine_q_mesh_only` flag was set to `True` in the `INPUTHP` namelist.
        In this case, there will only be a stdout file. All other output files will be missing, but that is expected.
        """
        return self.node.inputs.parameters.base.attributes.get('INPUTHP', {}).get('determine_q_mesh_only', False)

    @property
    def is_partial_site(self):
        """Return whether the calculation computed just a sub set of all sites to be perturbed.

        A complete run means that all perturbations were calculated and the final matrices were computed.
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

        :return: log messages
        """
        from .parse_raw.hp import parse_raw_output

        filename = self.node.base.attributes.get('output_filename')

        if filename not in self.retrieved.base.repository.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            stdout = self.retrieved.base.repository.get_object_content(filename)
        except IOError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        try:
            parsed_data, logs = parse_raw_output(stdout)
        except Exception:  # pylint: disable=broad-except
            return self.exit_codes.ERROR_OUTPUT_STDOUT_PARSE

        self.out('parameters', orm.Dict(parsed_data))

        # If the stdout was incomplete, most likely the job was interrupted before it could cleanly finish, so the
        # output files are most likely corrupt and cannot be restarted from
        if 'ERROR_OUTPUT_STDOUT_INCOMPLETE' in logs['error']:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE  # pylint: disable=attribute-defined-outside-init

        return logs

    def validate_premature_exit(self, logs):
        """Analyze problems that will cause a pre-mature termination of the calculation, controlled or not."""
        for exit_status in [
            'ERROR_OUT_OF_WALLTIME',
            'ERROR_INVALID_NAMELIST',
            'ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS',
            'ERROR_MISSING_PERTURBATION_FILE',
            'ERROR_CONVERGENCE_NOT_REACHED',
            'ERROR_COMPUTING_CHOLESKY',
            'ERROR_MISSING_CHI_MATRICES',
            'ERROR_INCOMPATIBLE_FFT_GRID',
            'ERROR_FERMI_SHIFT',
        ]:
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

            self.out('hubbard', orm.Dict(parsed_data['hubbard_U']))
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

    def parse_hubbard_dat(self, folder_path):
        """Parse the Hubbard parameters output file.

        :return: optional exit code in case of an error
        """
        from aiida_quantumespresso.common.hubbard import Hubbard
        from aiida_quantumespresso.utils.hubbard import HubbardUtils
        filename = HpCalculation.filename_output_hubbard_dat

        filepath = os.path.join(folder_path, filename)

        hubbard_structure = self.node.inputs.hubbard_structure.clone()

        intersites = None
        if 'settings' in self.node.inputs:
            if 'radial_analysis' in self.node.inputs.settings.get_dict():
                kwargs = self.node.inputs.settings.dict.radial_analysis
                intersites = HubbardUtils(hubbard_structure).get_intersites_list(**kwargs)

        hubbard_structure.clear_hubbard_parameters()
        hubbard_utils = HubbardUtils(hubbard_structure)
        hubbard_utils.parse_hubbard_dat(filepath=filepath)

        if intersites is None:
            self.out('hubbard_structure', hubbard_utils.hubbard_structure)
        else:
            hubbard_list = np.array(hubbard_utils.hubbard_structure.hubbard.to_list(), dtype='object')
            parsed_intersites = hubbard_list[:, [0, 2, 5]].tolist()
            selected_indices = []

            for i, intersite in enumerate(parsed_intersites):
                if intersite in intersites:
                    selected_indices.append(i)

            hubbard = Hubbard.from_list(hubbard_list[selected_indices])
            hubbard_structure.hubbard = hubbard
            self.out('hubbard_structure', hubbard_structure)

    def get_hubbard_structure(self):
        """Set in output an ``HubbardStructureData`` with standard Hubbard U formulation."""
        from copy import deepcopy

        hubbard_structure = deepcopy(self.node.inputs.hubbard_structure)
        hubbard_structure.clear_hubbard_parameters()

        hubbard_sites = self.outputs.hubbard.get_dict()['sites']

        for hubbard_site in hubbard_sites:
            index = int(hubbard_site['index'])
            manifold = hubbard_site['manifold']
            value = float(hubbard_site['value'])
            args = (index, manifold, index, manifold, value, (0, 0, 0), 'Ueff')
            hubbard_structure.append_hubbard_parameter(*args)

        self.out('hubbard_structure', hubbard_structure)

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
            matrix = np.array(self.parse_hubbard_matrix(matrix_data))
            result[matrix_name] = matrix

        return result

    def parse_hubbard_content(self, handle):
        """Parse the contents of the file {prefix}.Hubbard_parameters.dat as written by a HpCalculation.

        :param filepath: absolute filepath to the Hubbard_parameters.dat output file
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
                            'index': int(subdata[0]) - 1,  # QE indices start from 1
                            'type': int(subdata[1]),
                            'kind': subdata[2],
                            'spin': int(subdata[3]),
                            'new_type': int(subdata[4]),
                            'new_kind': subdata[5],
                            'manifold': subdata[6],
                            'value': float(subdata[7]),
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
        """Parse one of the matrices that are written to the {prefix}.Hubbard_parameters.dat files.

        Each matrix should be square of size N, which is given by the product of the number of q-points and the number
        of Hubbard species. Each matrix row is printed with a maximum number of 8 elements per line and each line is
        followed by an empty line. In the parsing of the data, we will use the empty line to detect the end of the
        current matrix row.

        :param data: a list of strings representing lines in the Hubbard_parameters.dat file of a certain matrix
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

        return np.array(matrix)
