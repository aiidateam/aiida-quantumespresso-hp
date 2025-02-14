# -*- coding: utf-8 -*-
"""`CalcJob` implementation for the hp.x code of Quantum ESPRESSO."""
from __future__ import annotations

import os

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.utils import classproperty
from aiida.plugins import CalculationFactory, DataFactory
from aiida_quantumespresso.calculations import CalcJob, _lowercase_dict, _uppercase_dict
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

from aiida_hubbard.utils.general import is_perturb_only_atom

PwCalculation = CalculationFactory('quantumespresso.pw')
HubbardStructureData = DataFactory('quantumespresso.hubbard_structure')


def validate_parent_scf(parent_scf, _):
    """Validate the `parent_scf` input.

    Make sure that it is created by a ``PwCalculation`` that was run with an ``HubbardStructureData``.
    """
    creator = parent_scf.creator

    if not creator:
        return f'could not determine the creator of {parent_scf}'

    if creator.process_class is not PwCalculation:
        return f'creator of `parent_scf` {creator} is not a `PwCalculation`'

    hubbard_structure = parent_scf.creator.inputs.structure
    if not isinstance(hubbard_structure, HubbardStructureData):
        return f'parent calculation {parent_scf} was not run with `HubbardStructureData`'


def validate_parent_hp(parent_hp, _):
    """Validate the `parent_hp` input.

    Each entry in the `parent_hp` mapping should be a retrieved folder of a `HpCalculation`.
    """
    for label, retrieved in parent_hp.items():
        creator = retrieved.creator

        if not creator:
            return f'could not determine the creator of {retrieved}'

        if creator.process_class is not HpCalculation:
            return f'creator of `parent_hp.{label}` {creator} is not a `HpCalculation`'


def validate_parameters(parameters, _):
    """Validate the `parameters` input."""
    result = _uppercase_dict(parameters.get_dict(), dict_name='parameters')
    result = {key: _lowercase_dict(value, dict_name=key) for key, value in result.items()}

    # Check that required namelists are present
    for namelist in HpCalculation.compulsory_namelists:
        if namelist not in result:
            return f'the required namelist `{namelist}` was not defined'

    # Check for presence of blocked keywords
    for namelist, flag in HpCalculation.blocked_keywords:
        if namelist in result and flag in result[namelist]:
            return f'explicit definition of flag `{flag}` in namelist `{namelist}` is not allowed'


def validate_qpoints(qpoints, _):
    """Validate the `qpoints` input."""
    try:
        _, offset = qpoints.get_kpoints_mesh()
    except AttributeError:
        return 'support for explicit qpoints is not implemented, only meshes'

    if any(i != 0. for i in offset):
        return 'support for qpoint meshes with non-zero offsets is not implemented'


def validate_inputs(inputs, _):
    """Validate inputs that depend on one another."""
    parameters = inputs['parameters'].get_dict().get('INPUTHP', {})

    compute_hp = parameters.get('compute_hp', False)
    determine_atom_only = parameters.get('determine_num_pert_only', False)
    determine_mesh_only = parameters.get('determine_q_mesh_only', False)
    perturb_only_atom = bool(is_perturb_only_atom(parameters))

    if compute_hp and 'parent_hp' not in inputs and 'hubbard_structure' not in inputs:
        return (
            'parameter `INPUTHP.compute_hp` is `True` but no parent folders '
            'defined in `parent_hp` or no `hubbard_structure` in inputs'
        )

    if (determine_atom_only or perturb_only_atom) and 'hubbard_structure' not in inputs:
        return (
            'parameter `INPUTHP.determine_num_pert_only` or `INPUTHP.perturb_only_atom` '
            'are `True`/`not None` but no `hubbard_structure` in inputs'
        )

    message = 'parameter `INPUTHP.determine_q_mesh_only` is `True` but {}'
    if determine_mesh_only:
        if determine_atom_only:
            return message.format('`INPUTHP.determine_num_pert_only` is `True` as well')
        if not perturb_only_atom:
            return message.format('`INPUTHP.perturb_only_atom` is not set')


class HpCalculation(CalcJob):
    """`CalcJob` implementation for the hp.x code of Quantum ESPRESSO."""

    # Keywords that cannot be set manually, only by the plugin
    blocked_keywords = [
        ('INPUTHP', 'iverbosity'),
        ('INPUTHP', 'prefix'),
        ('INPUTHP', 'outdir'),
        ('INPUTHP', 'nq1'),
        ('INPUTHP', 'nq2'),
        ('INPUTHP', 'nq3'),
    ]
    compulsory_namelists = ['INPUTHP']
    prefix = 'aiida'

    # Not using symlink of pw folder to allow multiple hp to run on top of the same folder
    _default_symlink_usage = False

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.inputs['metadata']['options']['input_filename'].default = f'{cls.prefix}.in'
        spec.inputs['metadata']['options']['output_filename'].default = f'{cls.prefix}.out'
        spec.inputs['metadata']['options']['parser_name'].default = 'quantumespresso.hp'
        spec.inputs['metadata']['options']['withmpi'].default = True
        spec.input('parameters', valid_type=orm.Dict, validator=validate_parameters,
            help='The input parameters for the namelists.')
        spec.input('qpoints', valid_type=orm.KpointsData, validator=validate_qpoints,
            help='The q-point grid on which to perform the perturbative calculation.')
        spec.input('settings', valid_type=orm.Dict, required=False,
            help='Optional node for special settings.')
        spec.input('parent_scf', valid_type=orm.RemoteData, validator=validate_parent_scf)
        spec.input_namespace('parent_hp', valid_type=orm.FolderData, validator=validate_parent_hp)
        spec.input('hubbard_structure', valid_type=HubbardStructureData, required=False)

        spec.output('parameters', valid_type=orm.Dict,
            help='')
        spec.output('hubbard_structure', valid_type=HubbardStructureData, required=False,
            help='``HubbardStructureData`` containing the new Hubbard parameters.')
        spec.output('hubbard', valid_type=orm.Dict, required=False,
            help='Parsed Hubbard parameters from the ``Hubbard_parameters.dat`` file.')
        spec.output('hubbard_chi', valid_type=orm.ArrayData, required=False,
            help='')
        spec.output('hubbard_matrices', valid_type=orm.ArrayData, required=False,
            help='')
        spec.inputs.validator = validate_inputs
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

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete
        spec.exit_code(300, 'ERROR_OUTPUT_FILES',
            message='Problems with one or more output files.')
        spec.exit_code(310, 'ERROR_OUTPUT_STDOUT_READ',
            message='The stdout output file could not be read.')
        spec.exit_code(311, 'ERROR_OUTPUT_STDOUT_PARSE',
            message='The stdout output file could not be parsed.')
        spec.exit_code(312, 'ERROR_OUTPUT_STDOUT_INCOMPLETE',
            message='The stdout output file was incomplete.')
        spec.exit_code(313, 'ERROR_HUBBARD_DAT',
            message='The `HUBBARD.dat` could not be parsed.')
        spec.exit_code(350, 'ERROR_INVALID_NAMELIST',
            message='The namelist in the input file contained invalid syntax and could not be parsed.')
        spec.exit_code(360, 'ERROR_MISSING_PERTURBATION_FILE',
            message='One of the required perturbation inputs files was not found.')
        spec.exit_code(365, 'ERROR_INCORRECT_ORDER_ATOMIC_POSITIONS',
            message='The atomic positions were not sorted with Hubbard sites first.')
        spec.exit_code(366, 'ERROR_FERMI_SHIFT',
            message=('The code failed due to Fermi shift, probably due to low energy cutoff '
                'or due to an incorrect treatment of an insulating state (i.e. no smearing shoudl be used).'))

        # Significant errors but calculation can be used to restart
        spec.exit_code(400, 'ERROR_OUT_OF_WALLTIME',
            message='The calculation stopped prematurely because it ran out of walltime.')
        spec.exit_code(410, 'ERROR_CONVERGENCE_NOT_REACHED',
            message='The electronic minimization cycle did not reach self-consistency.')
        spec.exit_code(462, 'ERROR_COMPUTING_CHOLESKY',
            message='The code failed during the cholesky factorization.')
        spec.exit_code(490, 'ERROR_MISSING_CHI_MATRICES',
            message='The code failed to reconstruct the full chi matrix as some chi matrices were missing')
        spec.exit_code(495, 'ERROR_INCOMPATIBLE_FFT_GRID',
            message='The code failed due to incompatibility between the FFT grid and the parallelization options.')
        # yapf: enable

    @classproperty
    def filename_output_hubbard_chi(cls):  # pylint: disable=no-self-argument
        """Return the relative output filename that contains chi."""
        return f'{cls.prefix}.chi.dat'

    @classproperty
    def filename_output_hubbard(cls):  # pylint: disable=no-self-argument
        """Return the relative output filename that contains the Hubbard values and matrices."""
        return f'{cls.prefix}.Hubbard_parameters.dat'

    @classproperty
    def filename_input_hubbard_parameters(cls):  # pylint: disable=no-self-argument,invalid-name
        """Return the relative input filename for Hubbard parameters, for QuantumESPRESSO version below 7.1."""
        return 'parameters.in'

    @classproperty
    def filename_output_hubbard_dat(cls):  # pylint: disable=no-self-argument,invalid-name
        """Return the relative input filename for generalised Hubbard parameters, for QuantumESPRESSO v.7.2 onwards."""
        return 'HUBBARD.dat'

    @classproperty
    def dirname_output(cls):  # pylint: disable=no-self-argument
        """Return the relative directory name that contains raw output data."""
        return 'out'

    @classproperty
    def dirname_output_hubbard(cls):  # pylint: disable=no-self-argument
        """Return the relative directory name that contains raw output data written by hp.x."""
        return os.path.join(cls.dirname_output, 'HP')

    @classproperty
    def dirname_output_scf(cls):  # pylint: disable=no-self-argument
        """Return the relative directory name that contains raw output data written by pw.x."""
        return os.path.join(cls.dirname_output, f'{cls.prefix}.save')

    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        if 'settings' in self.inputs:
            settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        else:
            settings = {}

        symlink = settings.pop('PARENT_FOLDER_SYMLINK', self._default_symlink_usage)  # a boolean

        parameters = self.prepare_parameters()
        self.write_input_files(folder, parameters)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params = (list(settings.pop('CMDLINE', [])) + ['-in', self.options.input_filename])

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = self.get_retrieve_list()
        # No need to keep ``HUBBARD.dat``, as the info is stored in ``aiida.Hubbard_parameters.dat``
        calcinfo.retrieve_temporary_list = [self.filename_output_hubbard_dat]
        if symlink:
            if 'parent_hp' not in self.inputs:
                folder.get_subfolder(self.dirname_output, create=True)
            calcinfo.remote_symlink_list = self.get_remote_copy_list(symlink)
        else:
            calcinfo.remote_copy_list = self.get_remote_copy_list(symlink)
        if 'parent_hp' in self.inputs:
            calcinfo.local_copy_list, calcinfo.provenance_exclude_list = self.get_local_copy_list()

        return calcinfo

    def get_retrieve_list(self) -> list[tuple]:
        """Return the `retrieve_list`.

        A `HpCalculation` can be parallelized over atoms by running individual calculations, but a final post-processing
        calculation will have to be performed to compute the final matrices. The final calculation that computes chi
        requires the perturbation files for all

        :returns: list of resource retrieval instructions
        """
        retrieve_list = []

        # Default output files that are written after a completed or post-processing HpCalculation
        retrieve_list.append(self.options.output_filename)
        retrieve_list.append(self.filename_output_hubbard)
        retrieve_list.append(self.filename_output_hubbard_dat)
        retrieve_list.append(os.path.join(self.dirname_output_hubbard, self.filename_output_hubbard_chi))

        # The perturbation files that are necessary for a final `compute_hp` calculation in case this is an incomplete
        # calculation that computes just a subset of all qpoints and/or all perturbed atoms.
        src_perturbation_files = os.path.join(self.dirname_output_hubbard, f'{self.prefix}.*.pert_*.dat')
        dst_perturbation_files = '.'
        retrieve_list.append((src_perturbation_files, dst_perturbation_files, 3))

        return retrieve_list

    def get_remote_copy_list(self, is_symlink) -> list[tuple]:
        """Return the `remote_{copy/symlink}_list`.

        :param is_symlink: whether to use symlink for the remote list
        :returns: list of resource copy instructions
        """
        parent_scf = self.inputs.parent_scf
        if 'parent_hp' in self.inputs and not is_symlink:
            dirname = self.dirname_output_scf
            dirfinal = self.dirname_output
        elif is_symlink:
            dirname = os.path.join(self.dirname_output, '*')
            dirfinal = self.dirname_output
        else:
            dirname = self.dirname_output
            dirfinal = '.'
        folder_src = os.path.join(parent_scf.get_remote_path(), dirname)
        return [(parent_scf.computer.uuid, folder_src, dirfinal)]

    def get_local_copy_list(self) -> tuple[list, list]:
        """Return the `local_copy_list`.

        :returns: tuple,list of resource copy instructions
        """
        local_copy_list, provenance_exclude_list = [], []

        for retrieved in self.inputs.get('parent_hp', {}).values():
            local_copy_list.append((retrieved.uuid, self.dirname_output_hubbard, self.dirname_output_hubbard))
            for filename in retrieved.base.repository.list_object_names(self.dirname_output_hubbard):
                filepath = os.path.join(self.dirname_output_hubbard, filename)
                provenance_exclude_list.append(filepath)

        return local_copy_list, provenance_exclude_list

    def prepare_parameters(self) -> dict:
        """Prepare the parameters based on the input parameters.

        The returned input dictionary will contain all the necessary namelists and their flags that should be written to
        the input file of the calculation.

        :returns: a dictionary with input namelists and their flags
        """
        result = _uppercase_dict(self.inputs.parameters.get_dict(), dict_name='parameters')
        result = {key: _lowercase_dict(value, dict_name=key) for key, value in result.items()}

        mesh, _ = self.inputs.qpoints.get_kpoints_mesh()

        if 'parent_hp' in self.inputs:
            atom_perturbed = bool(is_perturb_only_atom(result.get('INPUTHP', {})))
            # `sum_perq` and `compute_hp` can be used only separately
            if atom_perturbed:
                result['INPUTHP']['sum_pertq'] = True
            else:
                result['INPUTHP']['compute_hp'] = True

        result['INPUTHP']['iverbosity'] = 2
        result['INPUTHP']['outdir'] = self.dirname_output
        result['INPUTHP']['prefix'] = self.prefix
        result['INPUTHP']['nq1'] = mesh[0]
        result['INPUTHP']['nq2'] = mesh[1]
        result['INPUTHP']['nq3'] = mesh[2]

        return result

    def write_input_files(self, folder, parameters):
        """Write the prepared `parameters` to the input file in the sandbox folder.

        :param folder: an :class:`aiida.common.folders.Folder` to temporarily write files on disk.
        :param parameters: a dictionary with input namelists and their flags.
        """
        # Write the main input file
        with folder.open(self.options.input_filename, 'w') as handle:
            for namelist_name in self.compulsory_namelists:
                namelist = parameters.pop(namelist_name)
                handle.write(f'&{namelist_name}\n')
                for key, value in sorted(namelist.items()):
                    handle.write(convert_input_to_namelist_entry(key, value))
                handle.write('/\n')
