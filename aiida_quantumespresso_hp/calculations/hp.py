# -*- coding: utf-8 -*-
import os
from aiida.orm import CalculationFactory
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError, ValidationError, UniquenessError, NotExistent
from aiida.common.links import LinkType
from aiida.common.utils import classproperty
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.calculation.job import JobCalculation
from aiida.orm.calculation.function import FunctionCalculation
from aiida_quantumespresso.calculations import get_input_data_text, _lowercase_dict, _uppercase_dict


PwCalculation = CalculationFactory('quantumespresso.pw')


class HpCalculation(JobCalculation):
    """
    Quantum ESPRESSO Hp calculations
    """

    def _init_internal_params(self):
        super(HpCalculation, self)._init_internal_params()

        self._PREFIX = 'aiida'
        self._INPUT_FILE_NAME = '{}.in'.format(self._PREFIX)
        self._OUTPUT_FILE_NAME = '{}.out'.format(self._PREFIX)
        self._OUTPUT_CHI_SUFFIX = '.chi.dat'
        self._OUTPUT_HUBBARD_SUFFIX = '.Hubbard_parameters.dat'

        self._default_parser = 'quantumespresso.hp'
        self._compulsory_namelists = ['INPUTHP']
        self._optional_inputs = ['settings']
        self._required_inputs = ['code', 'parameters', 'parent_folder', 'qpoints']

        # Keywords that cannot be set manually, only by the plugin
        self._blocked_keywords = [
            ('INPUTHP', 'iverbosity'),
            ('INPUTHP', 'prefix'),
            ('INPUTHP', 'outdir'),
            ('INPUTHP', 'nq1'),
            ('INPUTHP', 'nq2'),
            ('INPUTHP', 'nq3'),
        ]

        # Default input and output files
        self._DEFAULT_INPUT_FILE = self._INPUT_FILE_NAME
        self._DEFAULT_OUTPUT_FILE = self._OUTPUT_FILE_NAME

    @classproperty
    def _OUTPUT_SUBFOLDER(cls):
        return './out/'

    @classproperty
    def _FOLDER_RAW(cls):
        return os.path.join(cls._OUTPUT_SUBFOLDER, 'HP')

    @classproperty
    def _use_methods(cls):
        """
        Additional use_* methods for the Hp calculation class
        """
        retdict = JobCalculation._use_methods
        retdict.update({
            'parameters': {
                'valid_types': (ParameterData),
                'additional_parameter': None,
                'linkname': 'parameters',
                'docstring': ('A node that specifies the input parameters for the namelists'),
            },
            'parent_folder': {
                'valid_types': (FolderData, RemoteData),
                'additional_parameter': None,
                'linkname': 'parent_folder',
                'docstring': ('The remote folder of a completed PwCalculation with lda_plus_u switch turned on'),
            },
            'qpoints': {
                'valid_types': (KpointsData),
                'additional_parameter': None,
                'linkname': 'qpoints',
                'docstring': ('Specify the q-point grid on which to perform the perturbative calculation'),
            },
            'settings': {
                'valid_types': (ParameterData),
                'additional_parameter': None,
                'linkname': 'settings',
                'docstring': ('An optional node for special settings'),
            },
        })
        return retdict

    def _get_input_valid_types(self, key):
        return self._use_methods[key]['valid_types']

    def _get_input_valid_type(self, key):
        valid_types = self._get_input_valid_types(key)

        if isinstance(valid_types, tuple):
            return valid_types[0]
        else:
            return valid_types

    def _prepare_for_submission(self, tempfolder, input_nodes_raw):        
        """
        This method is called prior to job submission with a set of calculation input nodes.
        The inputs will be validated and sanitized, after which the necessary input files will
        be written to disk in a temporary folder. A CalcInfo instance will be returned that contains
        lists of files that need to be copied to the remote machine before job submission, as well
        as file lists that are to be retrieved after job completion.

        :param tempfolder: an aiida.common.folders.Folder to temporarily write files on disk
        :param input_nodes_raw: a dictionary with the raw input nodes
        :returns: CalcInfo instance
        """
        input_nodes = self.validate_input_nodes(input_nodes_raw)
        input_parent_folder = self.validate_input_parent_folder(input_nodes)
        input_parameters = self.validate_input_parameters(input_nodes)
        input_settings = input_nodes[self.get_linkname('settings')].get_dict()
        input_code = input_nodes[self.get_linkname('code')]

        self.write_input_files(tempfolder, input_parameters)

        retrieve_list = self.get_retrieve_list(input_nodes)
        local_copy_list = self.get_local_copy_list(input_nodes)
        remote_copy_list = self.get_remote_copy_list(input_nodes)

        # Empty command line by default
        cmdline_params = input_settings.pop('CMDLINE', [])

        codeinfo = CodeInfo()
        codeinfo.cmdline_params = (list(cmdline_params) + ["-in", self._INPUT_FILE_NAME])
        codeinfo.stdout_name = self._OUTPUT_FILE_NAME
        codeinfo.code_uuid = input_code.uuid

        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = retrieve_list
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        return calcinfo

    def get_retrieve_list(self, input_nodes):
        """
        Build the list of files that are to be retrieved upon calculation completion so that they can
        be passed to the parser.

        A HpCalculation can be parallelized over atoms by running individual calculations, but a
        final post-processing calculation will have to be performed to compute the final matrices
        The current version of hp.x requires the following folders and files:

            * Perturbation files: by default in _FOLDER_RAW/_PREFIX.chi.pert_*.dat
            * QE save directory: by default in _OUTPUT_SUBFOLDER/_PREFIX.save
            * The occupations file: by default in _OUTPUT_SUBFOLDER/_PREFIX.occup

        :param input_nodes: dictionary of validated and sanitized input nodes
        :returns: list of resource retrieval instructions
        """
        retrieve_list = []

        # Default output files that are written after a completed or post-processing HpCalculation
        retrieve_list.append(self._OUTPUT_FILE_NAME)
        retrieve_list.append(self._PREFIX + self._OUTPUT_CHI_SUFFIX)
        retrieve_list.append(self._PREFIX + self._OUTPUT_HUBBARD_SUFFIX)

        # Required files and directories for final collection calculations
        path_save_directory = os.path.join(self._OUTPUT_SUBFOLDER, self._PREFIX + '.save')
        path_occup_file = os.path.join(self._OUTPUT_SUBFOLDER, self._PREFIX + '.occup')

        retrieve_list.append([path_save_directory, path_save_directory, 0])
        retrieve_list.append([path_occup_file, path_occup_file, 0])

        src_perturbation_files = os.path.join(self._FOLDER_RAW, '{}.chi.pert_*.dat'.format(self._PREFIX))
        dst_perturbation_files = '.'
        retrieve_list.append([src_perturbation_files, dst_perturbation_files, 3])

        return retrieve_list

    def get_local_copy_list(self, input_nodes):
        """
        Build the local copy list, which amounts to copying the output subfolder of the specified
        parent_folder input node if it is a local FolderData object

        :param input_nodes: dictionary of validated and sanitized input nodes
        :returns: list of resource copy instructions
        """
        local_copy_list = []
        parent_folder = input_nodes['parent_folder']

        if isinstance(parent_folder, FolderData):
            folder_src = parent_folder.get_abs_path(self._OUTPUT_SUBFOLDER)
            folder_dst = self._OUTPUT_SUBFOLDER
            local_copy_list.append((folder_src, folder_dst))

        return local_copy_list

    def get_remote_copy_list(self, input_nodes):
        """
        Build the remote copy list, which amounts to copying the output subfolder of the specified
        parent_folder input node if it is a remote RemoteData object

        :param input_nodes: dictionary of validated and sanitized input nodes
        :returns: list of resource copy instructions
        """
        remote_copy_list = []
        parent_folder = input_nodes['parent_folder']

        if isinstance(parent_folder, RemoteData):
            computer_uuid = parent_folder.get_computer().uuid
            folder_src = os.path.join(parent_folder.get_remote_path(), self._OUTPUT_SUBFOLDER)
            folder_dst = self._OUTPUT_SUBFOLDER
            remote_copy_list.append((computer_uuid, folder_src, folder_dst))

        return remote_copy_list

    def validate_input_nodes(self, input_nodes_raw):
        """
        This function will validate that all required input nodes are present and that their content
        is valid. The parameter node necessary to write the input will also be created

        :param input_nodes_raw: a dictionary with the raw input nodes
        :returns: dictionary with validated and sanitized input nodes
        """
        input_nodes = {}

        # Verify that all required inputs are provided in the raw input dictionary
        for input_key in self._required_inputs:
            try:
                input_link = self.get_linkname(input_key)
                input_node = input_nodes_raw.pop(input_key)
            except KeyError:
                raise InputValidationError("required input '{}' was not specified".format(input_key))

            input_nodes[input_link] = input_node

        # Check for optional inputs in the raw input dictionary, creating an instance of its valid types otherwise
        for input_key in self._optional_inputs:
            try:
                input_link = self.get_linkname(input_key)
                input_node = input_nodes_raw.pop(input_key)
            except KeyError:
                valid_types = self._use_methods[input_key]['valid_types']
                valid_type_class = self._get_input_valid_type(input_key)
                input_node = valid_type_class()

            input_nodes[input_link] = input_node

        # Any remaining input nodes are not recognized raise an input validation exception
        if input_nodes_raw:
            raise InputValidationError('the following input nodes were not recognized: {}'.format(input_nodes_raw.keys()))

        return input_nodes

    def validate_input_parent_folder(self, input_nodes):
        """
        Validate the parent_folder node from the verified input_nodes, making sure that it belongs to either a

            * PwCalculation that was run with the 'lda_plus_u' switch turned on
            * HpCalculation indicating a restart from unconverged calculation
            * FunctionCalculation which may be used in workchains for collecting chi matrix calculations

        :param input_nodes: dictionary of sanitized and validated input nodes
        :returns: the parent_folder input node if valid
        :raises: if the parent_folder does not correspond to PwCalculation with 'lda_plus_u'
        """
        parent_folder_link = self.get_linkname('parent_folder')
        parent_folder_node = input_nodes[parent_folder_link]

        parent_calculation = parent_folder_node.get_inputs(link_type=LinkType.CREATE)

        if len(parent_calculation) != 1:
            raise ValueError('could not determine the parent calculation of the parent_folder input node')

        parent_calculation = parent_calculation[0]

        if not isinstance(parent_calculation, (PwCalculation, HpCalculation, FunctionCalculation)):
            raise ValueError('the parent calculation of the parent folder input node is not a {}, {} or {}'
                .format(PwCalculation.__name__, HpCalculation.__name__, FunctionCalculation))

        if isinstance(parent_calculation, (HpCalculation, FunctionCalculation)):
            return parent_folder_node

        try:
            input_parameters = parent_calculation.inp.parameters
        except KeyError:
            raise ValueError('could not retrieve the input parameters node from the parent calculation')

        lda_plus_u = input_parameters.get_dict().get('SYSTEM', {}).get('lda_plus_u', False)

        if not lda_plus_u:
            raise ValueError("the input parameters<{}> of the parent calculation did not specify 'lda_plus_u: True'"
                .format(input_parameters.pk))

        return parent_folder_node

    def validate_input_parameters(self, input_nodes):
        """
        Validate the parameters input node and create from it the input parameter dictionary that contains
        all the necessary namelists and their flags that should be written to the input file of the calculation

        :param input_nodes: dictionary of sanitized and validated input nodes
        :returns: input_parameters a dictionary with input namelists and their flags
        """
        qpoints = input_nodes[self.get_linkname('qpoints')]
        parameters = input_nodes[self.get_linkname('parameters')].get_dict()

        # Transform first-level keys (i.e. namelist and card names) to uppercase and second-level to lowercase
        input_parameters = _uppercase_dict(parameters, dict_name='parameters')
        input_parameters = {k: _lowercase_dict(v, dict_name=k) for k, v in input_parameters.iteritems()}

        # Check that required namelists are present
        for namelist in self._compulsory_namelists:
            if not namelist in input_parameters:
                raise InputValidationError("the required namelist '{}' was not defined".format(namelist))

        # Check for presence of blocked keywords
        for namelist, flag in self._blocked_keywords:
            if namelist in input_parameters and flag in input_parameters[namelist]:
                raise InputValidationError("explicit definition of the '{}' "
                    "flag in the '{}' namelist or card is not allowed".format(flag, namelist))

        # Validate qpoint input node
        try:
            mesh, offset = qpoints.get_kpoints_mesh()
        except AttributeError:
            raise NotImplementedError("support for explicit qpoints is not implemented, only uniform meshes")

        if any([i != 0. for i in offset]):
            raise NotImplementedError('support for qpoint meshes with non-zero offsets is not implemented')

        input_parameters['INPUTHP']['iverbosity'] = 2
        input_parameters['INPUTHP']['outdir'] = self._OUTPUT_SUBFOLDER
        input_parameters['INPUTHP']['prefix'] = self._PREFIX
        input_parameters['INPUTHP']['nq1'] = mesh[0]
        input_parameters['INPUTHP']['nq2'] = mesh[1]
        input_parameters['INPUTHP']['nq3'] = mesh[2]

        return input_parameters

    def write_input_files(self, tempfolder, input_parameters):
        """
        Take the input_parameters dictionary with the namelists and their flags
        and write the input file to disk in the temporary folder

        :param tempfolder: an aiida.common.folders.Folder to temporarily write files on disk
        :param input_parameters: a dictionary with input namelists and their flags
        """
        filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)

        with open(filename, 'w') as handle:
            for namelist_name in self._compulsory_namelists:
                namelist = input_parameters.pop(namelist_name)
                handle.write("&{0}\n".format(namelist_name))
                for key, value in sorted(namelist.iteritems()):
                    handle.write(get_input_data_text(key, value))
                handle.write("/\n")

        if input_parameters:
            raise InputValidationError('these specified namelists are invalid: {}'
                .format(', '.join(input_parameters.keys())))
