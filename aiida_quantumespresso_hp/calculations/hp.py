# -*- coding: utf-8 -*-
"""`CalcJob` implementation for the hp.x code of Quantum ESPRESSO."""
from __future__ import absolute_import
import os
import six

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError
from aiida.common.utils import classproperty
from aiida.engine import CalcJob
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.calculations import _lowercase_dict, _uppercase_dict
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

PwCalculation = CalculationFactory('quantumespresso.pw')


class HpCalculation(CalcJob):
    """`CalcJob` implementation for the hp.x code of Quantum ESPRESSO."""

    # Keywords that cannot be set manually, only by the plugin
    _blocked_keywords = [
        ('INPUTHP', 'iverbosity'),
        ('INPUTHP', 'prefix'),
        ('INPUTHP', 'outdir'),
        ('INPUTHP', 'nq1'),
        ('INPUTHP', 'nq2'),
        ('INPUTHP', 'nq3'),
    ]
    _compulsory_namelists = ['INPUTHP']

    _prefix = 'aiida'
    _default_input_file = '{}.in'.format(_prefix)
    _default_output_file = '{}.out'.format(_prefix)
    _dirname_output = 'out'
    _filename_input_hubbard_parameters = 'parameters.in'
    _filename_output_hubbard_parameters = 'parameters.out'

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super(HpCalculation, cls).define(spec)
        spec.input('metadata.options.input_filename', valid_type=six.string_types, default=cls._default_input_file)
        spec.input('metadata.options.output_filename', valid_type=six.string_types, default=cls._default_output_file)
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='quantumespresso.hp')
        spec.input('parameters', valid_type=orm.Dict,
            help='The input parameters for the namelists.')
        spec.input('parent_folder', valid_type=(orm.FolderData, orm.RemoteData),
            help='The remote folder of a completed `PwCalculation` with `lda_plus_u` switch turned on')
        spec.input('qpoints', valid_type=orm.KpointsData,
            help='The q-point grid on which to perform the perturbative calculation.')
        spec.input('settings', valid_type=orm.Dict, required=False,
            help='Optional node for special settings.')
        spec.output('parameters', valid_type=orm.Dict,
            help='')
        spec.output('hubbard', valid_type=orm.Dict, required=False,
            help='')
        spec.output('hubbard_chi', valid_type=orm.ArrayData, required=False,
            help='')
        spec.output('hubbard_matrices', valid_type=orm.ArrayData, required=False,
            help='')
        spec.output('hubbard_parameters', valid_type=orm.SinglefileData, required=False,
            help='')
        spec.default_output_node = 'parameters'

        # Unrecoverable errors: resources like the retrieved folder or its expected contents are missing
        spec.exit_code(200, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.')
        spec.exit_code(210, 'ERROR_OUTPUT_STDOUT_MISSING',
            message='The retrieved folder did not contain the required stdout output file.')
        spec.exit_code(211, 'ERROR_OUTPUT_HUBBARD_MISSING',
            message='The retrieved folder did not contain the required hubbard output file.')
        spec.exit_code(212, 'ERROR_OUTPUT_HUBBARD_CHI_MISSING',
            message='The retrieved folder did not contain the required hubbard chi output file.')
        spec.exit_code(213, 'ERROR_OUTPUT_HUBBARD_PARAMETERS_MISSING',
            message='The retrieved folder did not contain the required hubbard parameters output file.')

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete
        spec.exit_code(300, 'ERROR_OUTPUT_FILES',
            message='Problems with one or more output files.')
        spec.exit_code(310, 'ERROR_OUTPUT_STDOUT_READ',
            message='The stdout output file could not be read.')
        spec.exit_code(311, 'ERROR_OUTPUT_STDOUT_PARSE',
            message='The stdout output file could not be parsed.')
        spec.exit_code(312, 'ERROR_OUTPUT_STDOUT_INCOMPLETE',
            message='The stdout output file was incomplete.')
        spec.exit_code(350, 'ERROR_INVALID_NAMELIST',
            message='The namelist in the input file contained invalid syntax and could not be parsed.')
        spec.exit_code(360, 'ERROR_MISSING_PERTURBATION_FILE',
            message='One of the required perturbation inputs files was not found.')
        spec.exit_code(365, 'ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS',
            message='The atomic positions were not sorted with Hubbard sites first.')

        # Significant errors but calculation can be used to restart
        spec.exit_code(400, 'ERROR_OUT_OF_WALLTIME',
            message='The calculation stopped prematurely because it ran out of walltime.')
        spec.exit_code(410, 'ERROR_CONVERGENCE_NOT_REACHED',
            message='The electronic minimization cycle did not reach self-consistency.')

    @classproperty
    def filename_output_hubbard_chi(cls):  # pylint: disable=no-self-argument
        """Return the relative output filename that contains chi."""
        return '{}.chi.dat'.format(cls._prefix)

    @classproperty
    def filename_output_hubbard(cls):  # pylint: disable=no-self-argument
        """Return the relative output filename that contains the Hubbard values and matrices."""
        return '{}.Hubbard_parameters.dat'.format(cls._prefix)

    @classproperty
    def filename_output_hubbard_parameters(cls):  # pylint: disable=no-self-argument,invalid-name
        """Return the relative output filename that all Hubbard parameters."""
        return cls._filename_output_hubbard_parameters

    @classproperty
    def filename_input_hubbard_parameters(cls):  # pylint: disable=no-self-argument,invalid-name
        """Return the relative input filename that all Hubbard parameters."""
        return cls._filename_input_hubbard_parameters

    @classproperty
    def dirname_output(cls):  # pylint: disable=no-self-argument
        """Return the relative directory name that contains raw output data."""
        return cls._dirname_output

    @classproperty
    def dirname_output_hubbard(cls):  # pylint: disable=no-self-argument
        """Return the relative directory name that contains raw output data written by hp.x."""
        return os.path.join(cls.dirname_output, 'HP')

    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        if 'settings' in self.inputs:
            settings = self.inputs.settings.get_dict()
        else:
            settings = {}

        parent_folder = self.validate_input_parent_folder(self.inputs.parent_folder)
        parameters = self.validate_input_parameters(self.inputs.parameters, self.inputs.qpoints)

        self.write_input_files(folder, parameters)

        cmdline_params = settings.pop('CMDLINE', [])  # Empty command line by default

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params = (list(cmdline_params) + ['-in', self.options.input_filename])

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = self.get_retrieve_list()
        calcinfo.local_copy_list = self.get_local_copy_list(parent_folder)
        calcinfo.remote_copy_list = self.get_remote_copy_list(parent_folder)

        return calcinfo

    def get_retrieve_list(self):
        """Return the `retrieve_list`.

        A `HpCalculation` can be parallelized over atoms by running individual calculations, but a final post-processing
        calculation will have to be performed to compute the final matrices. The current version of hp.x requires the
        following folders and files:

            * Perturbation files: by default in dirname_output_hubbard/_prefix.chi.pert_*.dat
            * QE save directory: by default in _dirname_output/_prefix.save
            * The occupations file: by default in _dirname_output/_prefix.occup

        :returns: list of resource retrieval instructions
        """
        retrieve_list = []

        # Default output files that are written after a completed or post-processing HpCalculation
        retrieve_list.append(self.options.output_filename)
        retrieve_list.append(self.filename_output_hubbard)
        retrieve_list.append(self.filename_output_hubbard_chi)
        retrieve_list.append(self.filename_output_hubbard_parameters)

        # Required files and directories for final collection calculations
        path_save_directory = os.path.join(self._dirname_output, self._prefix + '.save')
        path_occup_file = os.path.join(self._dirname_output, self._prefix + '.occup')
        path_paw_file = os.path.join(self._dirname_output, self._prefix + '.paw')

        retrieve_list.append([path_save_directory, path_save_directory, 0])
        retrieve_list.append([path_occup_file, path_occup_file, 0])
        retrieve_list.append([path_paw_file, path_paw_file, 0])

        src_perturbation_files = os.path.join(self.dirname_output_hubbard, '{}.chi.pert_*.dat'.format(self._prefix))
        dst_perturbation_files = '.'
        retrieve_list.append([src_perturbation_files, dst_perturbation_files, 3])

        return retrieve_list

    def get_local_copy_list(self, parent_folder):
        """Return the `local_copy_list`.

        Build the local copy list, which amounts to copying the output subfolder of the specified `parent_folder` input
        node if it is a local `FolderData` object.

        :param parent_folder: the `parent_folder` input node.
        :returns: list of resource copy instructions
        """
        local_copy_list = []

        if isinstance(parent_folder, orm.FolderData):
            local_copy_list.append((parent_folder.uuid, 'out', '.'))

        return local_copy_list

    def get_remote_copy_list(self, parent_folder):
        """Return the `remote_copy_list`.

        Build the remote copy list, which amounts to copying the output subfolder of the specified `parent_folder` input
        node if it is a remote `RemoteData` object.

        :param parent_folder: the `parent_folder` input node.
        :returns: list of resource copy instructions
        """
        remote_copy_list = []

        if isinstance(parent_folder, orm.RemoteData):
            computer_uuid = parent_folder.computer.uuid
            folder_src = os.path.join(parent_folder.get_remote_path(), self._dirname_output)
            folder_dst = self._dirname_output
            remote_copy_list.append((computer_uuid, folder_src, folder_dst))

        return remote_copy_list

    @staticmethod
    def validate_input_parent_folder(parent_folder):
        """Validate the `parent_folder` input.

        Make sure that it belongs to either a:

            * PwCalculation that was run with the `lda_plus_u` switch turned on
            * HpCalculation indicating a restart from unconverged calculation
            * WorkFunctionNode which may be used in workchains for collecting chi matrix calculations

        :param parent_folder: the `parent_folder` input node
        :returns: the `parent_folder` input node if valid
        :raises: if the `parent_folder` does not correspond to `PwCalculation` with `lda_plus_u=True`
        """
        creator = parent_folder.creator

        if not creator:
            raise ValueError('could not determine the creator of {}'.format(parent_folder))

        if isinstance(creator, orm.CalcJobNode) and creator.process_class not in {PwCalculation, HpCalculation}:
            raise ValueError('creator of `parent_folder` {} input node has incorrect type'.format(creator))

        if creator.process_class == PwCalculation:
            try:
                parameters = creator.inputs.parameters.get_dict()
            except KeyError:
                raise ValueError('could not retrieve the input parameters node from the parent calculation')

            lda_plus_u = parameters.get('SYSTEM', {}).get('lda_plus_u', False)

            if not lda_plus_u:
                raise ValueError('parent calculation {} was not run with `lda_plus_u`'.format(creator))

        return parent_folder

    def validate_input_parameters(self, parameters, qpoints):
        """Validate the parameters and qpoints input nodes and create from it the input parameter dictionary.

        The returned input dictionary will contain all the necessary namelists and their flags that should be written to
        the input file of the calculation.

        :param parameters: the `parameters` input node
        :param qpoints: the `qpoints` input node
        :returns: a dictionary with input namelists and their flags
        """
        # Transform first-level keys (i.e. namelist and card names) to uppercase and second-level to lowercase
        result = _uppercase_dict(parameters.get_dict(), dict_name='parameters')
        result = {key: _lowercase_dict(value, dict_name=key) for key, value in six.iteritems(result)}

        # Check that required namelists are present
        for namelist in self._compulsory_namelists:
            if namelist not in result:
                raise InputValidationError("the required namelist '{}' was not defined".format(namelist))

        # Check for presence of blocked keywords
        for namelist, flag in self._blocked_keywords:
            if namelist in result and flag in result[namelist]:
                raise InputValidationError(
                    "explicit definition of flag '{}' in namelist '{}' is not allowed".format(flag, namelist))

        try:
            mesh, offset = qpoints.get_kpoints_mesh()
        except AttributeError:
            raise NotImplementedError('support for explicit qpoints is not implemented, only uniform meshes')

        if any([i != 0. for i in offset]):
            raise NotImplementedError('support for qpoint meshes with non-zero offsets is not implemented')

        result['INPUTHP']['iverbosity'] = 2
        result['INPUTHP']['outdir'] = self._dirname_output
        result['INPUTHP']['prefix'] = self._prefix
        result['INPUTHP']['nq1'] = mesh[0]
        result['INPUTHP']['nq2'] = mesh[1]
        result['INPUTHP']['nq3'] = mesh[2]

        return result

    def write_input_files(self, folder, parameters):
        """Write the prepared `parameters` to the input file in the sandbox folder.

        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :param parameters: a dictionary with input namelists and their flags
        """
        with folder.open(self.options.input_filename, 'w') as handle:
            for namelist_name in self._compulsory_namelists:
                namelist = parameters.pop(namelist_name)
                handle.write(u'&{0}\n'.format(namelist_name))
                for key, value in sorted(six.iteritems(namelist)):
                    handle.write(convert_input_to_namelist_entry(key, value))
                handle.write(u'/\n')

        if parameters:
            invalid = ', '.join(list(parameters.keys()))
            raise InputValidationError('these specified namelists are invalid: {}'.format(invalid))
