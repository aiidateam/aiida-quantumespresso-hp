# -*- coding: utf-8 -*-
"""Turn-key solution to automatically compute the self-consistent Hubbard parameters for a given structure."""
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import WorkChain, ToContext, while_, if_, append_
from aiida.orm.nodes.data.array.bands import find_bandgap
from aiida.plugins import CalculationFactory, WorkflowFactory

from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults
from aiida_quantumespresso_hp.calculations.functions.structure_relabel_kinds import structure_relabel_kinds
from aiida_quantumespresso_hp.calculations.functions.structure_reorder_kinds import structure_reorder_kinds, structure_reorder_kinds_v_start
from aiida_quantumespresso_hp.calculations.functions.create_hubbard_v_from_distance import create_hubbard_v_from_distance
from aiida_quantumespresso_hp.utils.validation import validate_structure_kind_order
from aiida_quantumespresso_hp.utils.general import *


PwCalculation = CalculationFactory('quantumespresso.pw')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
HpWorkChain = WorkflowFactory('quantumespresso.hp.main')

# Sanity check
def validate_hubbard_file(hubbard_file, _): 
    """
    Validate the format of the hubbard file.
    Raises error if: 
       * the uncommented lines (data lines) are not composed of 3 colums, i.e. 3 values;
       * control format colums of data lines, i.e. each data line==[int, int, float]
       * comment lines are present between data lines
    
    ::TO BE CHECKED:: the last point could be useless; should check the accepted format of pw.x

    """
    with hubbard_file.open() as file:
        lines  = file.readlines()
        is_comment = True
        comment_lines_finished = False
        for line in lines:
            line_list = line.strip().split()
            if line_list[0] == '#':
                if is_comment == comment_lines_finished:
                    return 'bad file format; unexpected comment line after lines of data.'                
            else:
                comment_lines_finished = True
                if not len(line_list) == 3:
                    return 'bad file format; expecting 3 colums per row.'
                if not line_list[0].isdigit() and line_list[1].isdigit():
                    return 'one or more element in the first two colums of data are not integer numbers.'
                try:
                    float(line_list[2])
                except ValueError:
                    return 'one element in the third column of data cannot be converted in type float.'
    
def validate_inputs(inputs, _):
    """Validate the entire inputs namespace."""
    structure_kinds = inputs['structure'].get_kind_names()
    
    if 'hubbard_u' in inputs and ('hubbard_file' in inputs or 'hubbard_start' in inputs):
        return 'too many Hubbard inputs have been provided. Choose only one.'
    
    if not ('hubbard_u' in inputs or 'hubbard_file' in inputs or 'hubbard_start' in inputs):
        return 'no Hubbard input has been set. At least one needed.'
        
    if 'hubbard_u' in inputs:
        hubbard_u_kinds = list(inputs['hubbard_u'].get_dict().keys())
        
        if not hubbard_u_kinds:
            return 'need to define a starting Hubbard U value for at least one kind.'
        
        if not set(hubbard_u_kinds).issubset(structure_kinds):
            return 'kinds specified in starting Hubbard U values is not a strict subset of the structure kinds.'
        
    if 'hubbard_start' in inputs:
        hubbard_start = inputs['hubbard_start']
        
        for hubbard_atom in hubbard_start:
            if len(hubbard_atom)!=4:
                return 'a list of hubbard_start is not of the proper lenght; should be 4.'
            
            if ( type(hubbard_atom[0])!=str or type(hubbard_atom[1])!=str or type(hubbard_atom[2])!=int or 
                  ( type(hubbard_atom[3])!=float and type(hubbard_atom[3])!=int ) ):
                return 'a list of hubbard_start contains wrong type and/or type order; should be [str,str,int,float].'
            
            if not set(hubbard_atom[0:2]).issubset(structure_kinds):
                return 'kinds specified in a list of starting hubbard_start values is not a strict subset of the structure.'
            
            if hubbard_atom[2]!=1: # this condition could change in future versions of QE (>v6.7)
                return f'`k = {hubbard_atom[2]}` not implemented in hp.x.'
        
        
class SelfConsistentHubbardWorkChain(WorkChain):
    """
    Workchain that for a given input structure will compute the self-consistent Hubbard parameters
    by iteratively relaxing the structure with the PwRelaxWorkChain and computing the Hubbard
    parameters through the HpWorkChain, until the Hubbard values are converged within a certain tolerance(s).

    The procedure in each step of the convergence cycle is slightly different depending on the electronic and
    magnetic properties of the system. Each cycle will roughly consist of three steps:

        * Relaxing the structure at the current Hubbard values
        * One or more SCF calculations depending on the system's electronic and magnetic properties
        * A self-consistent calculation of the Hubbard parameters, restarted from the previous SCF run

    The possible options for the set of SCF calculations that have to be run in the second step look are:

        * Metals:
            - SCF with smearing

        * Non-magnetic insulators
            - SCF with fixed occupations

        * Magnetic insulators
            - SCF with smearing
            - SCF with fixed occupations, where total magnetization and number of bands are fixed
              to the values found from the previous SCF calculation

    When convergence is achieved a node will be returned containing the final converged
    Hubbard parameters.
    
    Available Self Consistent Hubbard calculations are:
    
        * U only:
             - activated providing in input `hubbard_u` only
        * U+V:
             - activated providing in input `hubbard_file` or `hubbard_start`
             
    Activating both `U only` and `U+V` will make the WorkChain stopped. Choose only one.
    Moreover, `hubbard_file` has priority on `hubbard_start` (the latter will be ignored).
    """

    defaults = AttributeDict({
        'qe': qe_defaults,
        'smearing_method': 'marzari-vanderbilt',
        'smearing_degauss': 0.01,
        'conv_thr_preconverge': 1E-10,
        'conv_thr_strictfinal': 1E-15,
        'u_projection_type_relax': 'ortho-atomic',
        'u_projection_type_scf': 'ortho-atomic',
    })

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Iputs
        spec.input('structure', valid_type=orm.StructureData)
        # .. hubbard U only
        spec.input('hubbard_u', valid_type=orm.Dict, required=False)
        # .. hubbard U+V
        spec.input('hubbard_file', valid_type=orm.SinglefileData, required=False,  validator=validate_hubbard_file,
                   help='A SinglefileData node with the Hubbard parameters from a previous hp.x calculation. ')
        spec.input('hubbard_start', valid_type=orm.List, required=False,
                   help='A List (of lists) node to activate the Hubbard U+V atoms, of the type '\
                   '[[str, str, int, float],...] <--> [[atom_kind_name_a, atom_kind_name_b, k, value],...]. '\
                   'It will automatically set the atomic position indices in the pw parameters '\
                   '( [na,nb,k,value] as in the QE doc), thus avoiding to look into the StructureData node. '\
                   'Overrides the input of the pw calculations, but not the hubbard_file inputs.')
        # .. parameters for self consistency
        spec.input('tolerance_u', valid_type=orm.Float, default=lambda: orm.Float(0.1),
                   help='Tolerance value for self-consistent calculation of Hubbard U. '
                   'In case of DFT+U+V calculation, it refers to the diagonal elements (i.e. on-site).')
        spec.input('tolerance_v', valid_type=orm.Float, default=lambda: orm.Float(0.01),
                   help='Tolerance value for self-consistent DFT+U+V calculation. '\
                   'It refers to the only off-diagonal elements V.')
        spec.input('cut_on_file', valid_type=orm.Float, required=False,
                   help='Cut value on hubbard_file. Meant to be used with small values to cut '\
                   'negligible (or negative) inter-site Hubbard parameters in the hubbard_file on each iteration.')
        spec.input('skip_first_relax', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                   help='if True, skip the first relaxation')
        spec.input('max_iterations', valid_type=orm.Int, default=lambda: orm.Int(5))
        spec.input('meta_convergence', valid_type=orm.Bool, default=lambda: orm.Bool(False))
        spec.inputs.validator = validate_inputs
        # .. workflows used
        spec.expose_inputs(PwRelaxWorkChain, namespace='relax', exclude=('structure',),
            namespace_options={'required': False, 'populate_defaults': False,
            'help': 'Inputs for the `PwRelaxWorkChain` that, when defined, will iteratively relax the structure.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='scf', exclude=('pw.structure',))
        spec.expose_inputs(HpWorkChain, namespace='hubbard', exclude=('hp.parent_scf',))
        
        # Outline of the workflow
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            while_(cls.should_run_iteration)(
                cls.update_iteration,
                if_(cls.should_run_relax)(
                    cls.run_relax,
                    cls.inspect_relax,
                ),
                cls.run_scf_smearing,
                cls.recon_scf,
                if_(cls.is_insulator)(
                    cls.run_scf_fixed,
                ),
                cls.inspect_scf,
                cls.run_hp,
                cls.inspect_hp,
            ),
            cls.run_results,
        )
        
        # Outputs
        spec.output('structure', valid_type=orm.StructureData, required=False,
            help='The final relaxed structure, only if relax inputs were defined.')
        spec.output('hubbard', valid_type=orm.Dict, required=False,
            help='The final converged Hubbard U parameters.')
        spec.output('hubbard_parameters', valid_type=orm.SinglefileData, required=False,
            help='The final converged Hubbard U and V parameters.')
        
        # Exit codes
        spec.exit_code(330, 'ERROR_FAILED_TO_DETERMINE_PSEUDO_POTENTIAL',
            message='Failed to determine the correct pseudo potential after the structure changed its kind names.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RECON',
            message='The reconnaissance PwBaseWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='The PwRelaxWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The scf PwBaseWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_HP',
            message='The HpWorkChain sub process failed in iteration {iteration}')
        spec.exit_code(405, 'ERROR_NON_INTEGER_TOT_MAGNETIZATION',
            message='The scf PwBaseWorkChain sub process in iteration {iteration}'\
                    'returned a non integer total magnetization (threshold exceeded).')
        spec.exit_code(406, 'ERROR_NON_COMPARABLE_FILES',
            message='The previous hubbard file in iteration {iteration} is not a subset of the current one (cannot compare).')
        
    def setup(self):
        """Set up the context."""
        self.ctx.current_structure = self.inputs.structure
        
        if 'hubbard_u' in self.inputs:
            self.ctx.current_hubbard_u = self.inputs.hubbard_u.get_dict()
        elif 'hubbard_file' in self.inputs:
            self.ctx.current_hubbard_file = self.inputs.hubbard_file
            self.ctx.has_hubbard_file = True
        else:
            self.ctx.current_hubbard_file = None
            self.ctx.has_hubbard_file = False
            self.ctx.hubbard_start_ = create_hubbard_v_from_distance(self.inputs.hubbard_start,
                                                                     self.ctx.current_structure).get_list()
        
        self.ctx.max_iterations = self.inputs.max_iterations.value
        self.ctx.current_magnetic_moments = None # starting_magnetization dict for collinear spin calcs
        self.ctx.is_converged = False
        self.ctx.is_magnetic = None
        self.ctx.is_insulator = None
        self.ctx.iteration = 0
        self.ctx.is_U = False
        self.ctx.is_U_V = False
        self.ctx.skip_first_relax = self.inputs.skip_first_relax
        
    def validate_inputs(self):
        """Validate inputs."""
        structure = self.inputs.structure
        
        # Hubbard validation 
        # .. U only
        if 'hubbard_u' in self.inputs:
            self.ctx.is_U = True
            hubbard_u = self.inputs.hubbard_u      
            try:
                validate_structure_kind_order(structure, list(hubbard_u.get_dict().keys()))
            except ValueError:
                self.report('structure has incorrect kind order, reordering...')
                self.ctx.current_structure = structure_reorder_kinds(structure, hubbard_u)
        # .. U+V 
        elif 'hubbard_start' in self.inputs:
            try:
                hubbard_kinds = []
                for array in self.inputs.hubbard_start.get_list():
                    hubbard_kinds.append(array[0])
                    hubbard_kinds.append(array[1])
                hubbard_kinds = list(set(hubbard_kinds))
                validate_structure_kind_order(structure, hubbard_kinds)
            except ValueError:
                hubbard_start = self.inputs.hubbard_start
                self.report('structure has incorrect kind order, reordering...')
                self.ctx.current_structure = structure_reorder_kinds_v_start(structure, hubbard_start)
                self.ctx.hubbard_start_ = create_hubbard_v_from_distance(hubbard_start,
                                                                     self.ctx.current_structure).get_list()
            self.ctx.is_U_V = True
        elif 'hubbard_file' in self.inputs:
            self.ctx.is_U_V = True

        if 'hubbard_start' in self.inputs and 'hubbard_file' in self.inputs:
            self.report('both `hubbard_start` and `hubbard_file` have been specified. '\
                        'Ignoring `hubbard_start` and continuing...')
            
        # Determine whether the system is to be treated as magnetic
        parameters = self.inputs.scf.pw.parameters.get_dict()
        nspin      = parameters.get('SYSTEM', {}).get('nspin', self.defaults.qe.nspin)
        if  nspin != 1:
            self.report('system is treated to be magnetic because `nspin != 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = True
            if nspin == 2:
                self.ctx.current_magnetic_moments = orm.Dict(dict=parameters.get('SYSTEM', {}).get('starting_magnetization') )
                if self.ctx.current_magnetic_moments == None:
                    raise NameError('Missing `starting_magnetization` input in `scf.pw.parameters` while `nspin == 2`.')
            else: 
                # Luca Binci (EPFL) is working on this implementation. No current version (qe v. <= 6.7) 
                # supports the non collinear case, i.e. nspin=4 
                raise NotImplementedError(f'nspin=`{nspin}` is not implemented in the hp.x code.')
        else:
            self.report('system is treated to be non-magnetic because `nspin == 1` in `scf.pw.parameters` input.')
            self.ctx.is_magnetic = False
            
    def should_run_relax(self):
        """Return whether a relax calculation needs to be run, which is true if `relax` is specified in inputs."""
        if self.ctx.skip_first_relax:
            self.ctx.skip_first_relax = False # only the first one will be skipped
            self.report('skip_first_relax has been set True; skipping first relaxion...')
            return False
        else:
            return 'relax' in self.inputs

    def should_run_iteration(self):
        """Return whether a new process should be run.

        This is the case as long as the Hubbard parameters have not yet converged and the maximum number of restarts has
        not yet been exceeded.
        """
        return not self.ctx.is_converged and self.ctx.iteration < self.ctx.max_iterations

    def update_iteration(self):
        """Update the current iteration index counter."""
        self.ctx.iteration += 1
    
    def is_U(self):
        """Return whether the current structure is a metal."""
        return self.ctx.is_U
 
    def is_U_V(self):
        """Return whether the current structure is a metal."""
        return self.ctx.is_U_V
    
    def has_hubbard_file(self):
        """Return whether a parameter file has been provided"""
        return self.ctx.has_hubbard_file
    
    def is_insulator(self):
        """Return whether the current structure is a metal."""
        return self.ctx.is_insulator

    def is_magnetic(self):
        """Return whether the current structure is magnetic."""
        return self.ctx.is_magnetic
    
    def get_inputs(self, cls, namespace):
        """Return the inputs for one of the subprocesses whose inputs are exposed in the given namespace.

        :param cls: the process class of the subprocess
        :param namespace: namespace into which the inputs are exposed.
        :return: dictionary with inputs.
        """
        inputs = AttributeDict(self.exposed_inputs(cls, namespace=namespace))

        try:
            pseudos = self.get_pseudos()
        except ValueError:
            return self.exit_codes.ERROR_FAILED_TO_DETERMINE_PSEUDO_POTENTIAL
        
        # !!!
        # Should define a new method to setup the inputs in order to have only one
        # which encapsulate the different Hubbard cases, i.e. U only, U+V (and possible future U+J, U+V+J)
        # !!!
        
        # Scf
        if cls is PwBaseWorkChain and namespace == 'scf':
            inputs.pw.pseudos = pseudos
            inputs.pw.structure = self.ctx.current_structure
            # general params
            inputs.pw.parameters = inputs.pw.parameters.get_dict()
            inputs.pw.parameters.setdefault('CONTROL', {})
            inputs.pw.parameters.setdefault('SYSTEM', {})
            inputs.pw.parameters.setdefault('ELECTRONS', {})
            # hubbard params
            if self.is_U():
                inputs.pw.parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
                inputs.pw.parameters['SYSTEM']['lda_plus_u'] = True
                inputs.pw.parameters['SYSTEM']['lda_plus_u_kind'] = 0
            if self.is_U_V():
                inputs.pw.parameters['SYSTEM']['lda_plus_u'] = True
                inputs.pw.parameters['SYSTEM']['lda_plus_u_kind'] = 2
                if self.has_hubbard_file():
                    inputs.pw.parameters['SYSTEM'].pop('hubbard_v', None)
                    inputs.pw.parameters['SYSTEM']['Hubbard_parameters'] = 'file'
                    inputs.pw.hubbard_file = self.ctx.current_hubbard_file
                else:
                    inputs.pw.parameters['SYSTEM']['hubbard_v'] = self.ctx.hubbard_start_
            # magnetic params
            if self.ctx.current_magnetic_moments != None:
                inputs.pw.parameters['SYSTEM']['starting_magnetization'] = self.ctx.current_magnetic_moments.get_dict()
        # Relax
        elif cls is PwRelaxWorkChain and namespace == 'relax':
            inputs.structure = self.ctx.current_structure
            inputs.base.pw.pseudos = pseudos
            # general params
            inputs.base.pw.parameters = inputs.base.pw.parameters.get_dict()
            inputs.base.pw.parameters.setdefault('CONTROL', {})
            inputs.base.pw.parameters.setdefault('SYSTEM', {})
            inputs.base.pw.parameters.setdefault('ELECTRONS', {})
            # hubbard params
            if self.is_U():
                inputs.base.pw.parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
                inputs.base.pw.parameters['SYSTEM']['lda_plus_u'] = True
                inputs.base.pw.parameters['SYSTEM']['lda_plus_u_kind'] = 0
            if self.is_U_V():
                inputs.base.pw.parameters['SYSTEM']['lda_plus_u'] = True
                inputs.base.pw.parameters['SYSTEM']['lda_plus_u_kind'] = 2
                if self.has_hubbard_file():
                    inputs.base.pw.parameters['SYSTEM'].pop('hubbard_v', None)
                    inputs.base.pw.parameters['SYSTEM']['Hubbard_parameters'] = 'file'
                    inputs.base.pw.hubbard_file = self.ctx.current_hubbard_file
                else:
                    inputs.base.pw.parameters['SYSTEM']['hubbard_v'] = self.ctx.hubbard_start_
            # magnetic params
            if self.ctx.current_magnetic_moments != None:
                inputs.base.pw.parameters['SYSTEM']['starting_magnetization'] = self.ctx.current_magnetic_moments.get_dict()
            # no final scf - would be time wasted
            inputs.pop('base_final_scf', None)
            inputs.final_scf = orm.Bool(False)

        return inputs
    
    def get_pseudos(self):
        """Return the mapping of pseudos based on the current structure.

        .. note:: this is necessary because during the workchain the kind names of the structure can change, meaning the
            mapping of the pseudos that is to be passed to the subprocesses also may have to change, since the keys are
            based on the kind names of the structure.

        :return: dictionary of pseudos where the keys are the kindnames of ``self.ctx.current_structure``.
        """
        import re

        results = {}
        pseudos = self.inputs.scf.pw.pseudos

        for kind in self.ctx.current_structure.kinds:
            for key, pseudo in pseudos.items():
                symbol = re.sub(r'\d', '', key)
                if re.match(r'{}[0-9]*'.format(kind.symbol), symbol):
                    results[kind.name] = pseudo
                    break
            else:
                raise ValueError(f'could not find the pseudo from inputs.scf.pw.pseudos for kind `{kind}`.')

        return results
    
    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation."""
        inputs = self.get_inputs(PwRelaxWorkChain, 'relax')
        parameters = inputs.base.pw.parameters
        
        u_projection_type_relax = parameters['SYSTEM'].get('u_projection_type', self.defaults.u_projection_type_relax)

        parameters['SYSTEM']['u_projection_type'] = self.defaults.u_projection_type_relax
        inputs.base.pw.parameters = orm.Dict(dict=parameters)
        inputs.metadata.call_link_label = 'iteration_{:02d}_relax'.format(self.ctx.iteration)

        if u_projection_type_relax != self.defaults.u_projection_type_relax:
            self.report(
                f'WARNING: you specified `u_projection_type = {u_projection_type_relax}` in the input parameters, but '
                r'this will crash pw.x, changing it to `{self.defaults.u_projection_type_relax}`'
            )

        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching PwRelaxWorkChain<{running.pk}> iteration #{self.ctx.iteration}')
        return ToContext(workchains_relax=append_(running))

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
        workchain = self.ctx.workchains_relax[-1]

        if not workchain.is_finished_ok:
            self.report(f'PwRelaxWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX.format(iteration=self.ctx.iteration)

        self.ctx.current_structure = workchain.outputs.output_structure

    def run_scf_smearing(self):
        """Run an scf `PwBaseWorkChain` with smeared occupations"""
        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'
        inputs.pw.parameters['SYSTEM']['occupations'] = 'smearing'
        inputs.pw.parameters['SYSTEM']['smearing'] = inputs.pw.parameters['SYSTEM'].get(
            'smearing', self.defaults.smearing_method
        )
        inputs.pw.parameters['SYSTEM']['degauss'] = inputs.pw.parameters['SYSTEM'].get(
            'degauss', self.defaults.smearing_degauss
        )
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = inputs.pw.parameters['SYSTEM'].get(
            'u_projection_type', self.defaults.u_projection_type_scf
        )
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = inputs.pw.parameters['ELECTRONS'].get(
            'conv_thr', self.defaults.conv_thr_preconverge
        )
        inputs.metadata.call_link_label = 'iteration_{:02d}_scf_smearing'.format(self.ctx.iteration)
        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(f'launching PwBaseWorkChain<{running.pk}> with smeared occupations')
        return ToContext(workchains_scf=append_(running))

    def run_scf_fixed(self):
        """
        Run an scf `PwBaseWorkChain` with fixed occupations on top of the previous calculation.
        The nunmber of bands and total magnetization (if magnetic) are set according to those of the 
        previous calculation that was run with smeared occupations.
        """
        previous_workchain = self.ctx.workchains_scf[-1]
        previous_parameters = previous_workchain.outputs.output_parameters

        inputs = self.get_inputs(PwBaseWorkChain, 'scf')
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'
        inputs.pw.parameters['CONTROL']['restart_mode'] = 'from_scratch'
        inputs.pw.parameters['SYSTEM']['occupations'] = 'fixed'
        inputs.pw.parameters['SYSTEM'].pop('degauss', None)
        inputs.pw.parameters['SYSTEM'].pop('smearing', None)
        inputs.pw.parameters['SYSTEM'].pop('starting_magnetization', None)
        inputs.pw.parameters['SYSTEM']['nbnd'] = previous_parameters.get_dict()['number_of_bands']
        
        # if magnetic, set the total magnetization and raises an error if is non (not close enough) integer
        if self.ctx.is_magnetic:
            if set_tot_magnetization( inputs.pw.parameters, previous_parameters.get_dict()['total_magnetization'] ):
                return self.exit_codes.ERROR_NON_INTEGER_TOT_MAGNETIZATION.format(iteration=self.ctx.iteration)
        
        u_proj_type = inputs.pw.parameters['SYSTEM'].get('u_projection_type', self.defaults.u_projection_type_scf)
        inputs.pw.parameters['SYSTEM']['u_projection_type'] = u_proj_type 
        conv_thr   = inputs.pw.parameters['ELECTRONS'].get('conv_thr', self.defaults.conv_thr_strictfinal)
        inputs.pw.parameters['ELECTRONS']['conv_thr'] = conv_thr
        
        # restarting from already converged charge density of previous smeared scf;
        # here current aiida-qe version does not allow to restart only from the pot. 
        inputs.pw.parent_folder = previous_workchain.outputs.remote_folder
        inputs.pw.parameters['ELECTRONS']['startingpot'] = 'file'
        
        inputs.pw.parameters = orm.Dict(dict=inputs.pw.parameters)
        
        if self.ctx.is_magnetic:
            inputs.metadata.call_link_label = 'iteration_{:02d}_scf_fixed_magnetic'.format(self.ctx.iteration)
            running = self.submit(PwBaseWorkChain, **inputs)
            self.report(f'launching PwBaseWorkChain<{running.pk}> with fixed occupations, '
                        r'bands and total magnetization')
        else:
            inputs.metadata.call_link_label = 'iteration_{:02d}_scf_fixed'.format(self.ctx.iteration)
            running = self.submit(PwBaseWorkChain, **inputs)
            self.report(f'launching PwBaseWorkChain<{running.pk}> with fixed occupations')
            
        return ToContext(workchains_scf=append_(running))

    def inspect_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchains_scf[-1]

        if not workchain.is_finished_ok:
            self.report(f'scf in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF.format(iteration=self.ctx.iteration)
        
    def recon_scf(self):
        """Verify that the scf PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchains_scf[-1]

        if not workchain.is_finished_ok:
            self.report(f'scf in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF.format(iteration=self.ctx.iteration)
        
        bands = workchain.outputs.output_band
        parameters = workchain.outputs.output_parameters.get_dict()
        number_electrons = parameters['number_of_electrons']

        is_insulator, _ = find_bandgap(bands, number_electrons=number_electrons)

        if is_insulator:
            self.report('after relaxation, system is determined to be an insulator')
            self.ctx.is_insulator = True
        else:
            self.report('after relaxation, system is determined to be a metal')
            self.ctx.is_insulator = False

    def run_hp(self):
        """
        Run the HpWorkChain restarting from the last completed scf calculation
        """
        workchain = self.ctx.workchains_scf[-1]

        inputs = self.get_inputs(HpWorkChain, 'hubbard')
        inputs.hp.parent_scf = workchain.outputs.remote_folder
        inputs.metadata.call_link_label = 'iteration_{:02d}_hp'.format(self.ctx.iteration)

        running = self.submit(HpWorkChain, **inputs)

        self.report(f'launching HpWorkChain<{running.pk}> iteration #{self.ctx.iteration}')
        return ToContext(workchains_hp=append_(running))

    def inspect_hp(self):
        """
        Analyze the last completed HpWorkChain. We check the current Hubbard parameters and compare those with
        the values computed in the previous iteration. If the difference for all Hubbard sites is smaller than
        the tolerance(s), the calculation is considered to be converged.
        """
        workchain = self.ctx.workchains_hp[-1]

        if not workchain.is_finished_ok:
            self.report(f'hp.x in iteration {self.ctx.iteration} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_HP.format(iteration=self.ctx.iteration)

        if not self.inputs.meta_convergence:
            self.report('meta convergence is switched off, so not checking convergence of Hubbard parameters.')
            self.ctx.is_converged = True
            return

        # U only
        if self.is_U():
            self.check_convergence_U(workchain)
        # U+V
        if self.is_U_V():
            if not self.has_hubbard_file():
                self.ctx.has_hubbard_file = True
                self.ctx.current_hubbard_file = workchain.outputs.hubbard_parameters
            else: 
                self.check_convergence_U_V(workchain) 
                self.ctx.current_hubbard_file = workchain.outputs.hubbard_parameters
                    
    def check_convergence_U(self, workchain):
        """Check the convergence of the U only case parameters"""
        prev_hubbard_u = self.ctx.current_hubbard_u

        # First check if new types were created, in which case we will have to create a new `StructureData`
        for site in workchain.outputs.hubbard.get_attribute('sites'):
            if site['type'] != site['new_type']:
                self.report('new types have been determined: relabeling the structure and starting new iteration.')
                result = structure_relabel_kinds(self.ctx.current_structure, 
                                                 workchain.outputs.hubbard, 
                                                 self.ctx.current_magnetic_moments)
                self.ctx.current_structure = result['structure']
                self.ctx.current_hubbard_u = result['hubbard_u'].get_dict()
                self.ctx.current_magnetic_moments = result['starting_magnetization']
                break
        else:
            self.ctx.current_hubbard_u = {}
            for entry in workchain.outputs.hubbard.get_dict()['sites']:
                self.ctx.current_hubbard_u[entry['kind']] = float(entry['value'])

        # Check per site if the new computed value is converged with respect to the last iteration
        for entry in workchain.outputs.hubbard.get_attribute('sites'):
            kind = entry['kind']
            index = entry['index']
            tolerance = self.inputs.tolerance_u.value
            current_value = float(entry['value'])
            previous_value = float(prev_hubbard_u[kind])
            if abs(current_value - previous_value) > tolerance:
                msg = f'parameters not converged for site {index}: {current_value} - {previous_value} > {tolerance}'
                self.report(msg)
                break
        else:
            self.report('Hubbard U parameters are converged.')
            self.ctx.is_converged = True                
     
    def check_convergence_U_V(self, workchain):
        """Check the convergence of the U only case parameters""" 

        # 1. Generate in - out (prev - curr) arrays w/o commented lines
        
        # .. parameters.in
        with self.ctx.current_hubbard_file.open() as prev_hubbard_file: 
            lines  = prev_hubbard_file.readlines()
            previous = []
            for line in lines:
                if line.strip().split()[0] != '#': # filtering comments
                    previous.append([x for x in line.strip().split()])
                   
        # .. parameters.out
        with workchain.outputs.hubbard_parameters.open() as curr_hubbard_file: 
            lines  = curr_hubbard_file.readlines()
            current = []
            for line in lines:
                if line.strip().split()[0] != '#': # filtering comments
                    current.append([x for x in line.strip().split()])
                   
        # .. arrays with only hubbard atoms indices
        check_prev = []
        for array in previous:
            check_prev.append(array[0:2])
        
        check_curr = []
        for array in current:
            check_curr.append(array[0:2])

        # 2. First we check the on-site convergence only.
        #   (this is a good idea since for the first iterations the indices on 
        #    the inter-site parameters may slightly change due to the relaxation)
        tolerance_u = self.inputs.tolerance_u.value                   
        on_site_converged = True

        for array in check_prev:
            index_curr = check_curr.index(array)
            index_prev = check_prev.index(array)
            previous_value = float(previous[index_prev][2])
            current_value  = float(current[index_curr][2])          
            if previous[index_prev][0]==previous[index_prev][1]:
                if abs(previous_value-current_value)>tolerance_u:
                    on_site_converged = False
                    msg = f'parameters not converged for {array}: {current_value} - {previous_value} > {tolerance_u}'
                    self.report(msg)
                    break

        # 3. Check convergence on inter-site
        inter_site_converged = True
        tolerance_v = self.inputs.tolerance_v.value

        if on_site_converged: # check inter-site convergence              
            for array in check_prev:
                index_curr = check_curr.index(array)
                index_prev = check_prev.index(array)
                previous_value = float(previous[index_prev][2])
                current_value  = float(current[index_curr][2])          
                if previous[index_prev][0]!=previous[index_prev][1]:
                    if abs(previous_value-current_value)>tolerance_v:
                        inter_site_converged = False
                        msg = f'parameters not converged for {array}: {current_value} - {previous_value} > {tolerance_v}'
                        self.report(msg)
                        break

        # 4. We check if the .in and .out are compatible at convergence, i.e. if the prev is a subset of the curr
        if on_site_converged and inter_site_converged:                   
            for array in check_prev:
                if not array in check_curr:
                    self.report('Convergence has been reached for the on-site parameters and '
                                'for a subset of the inter-site parameters, but Hubbard files do not match.')
                    return self.exit_codes.ERROR_NON_COMPARABLE_FILES.format(iteration=self.ctx.iteration)
            else:
                self.report('Hubbard U and V parameters are converged.')
                self.ctx.is_converged = True         
        
    
    def run_results(self):
        """Attach the final converged Hubbard U parameters and the corresponding structure."""
        self.report(f'Hubbard U parameters self-consistently converged in {self.ctx.iteration} iterations')
        self.out('structure', self.ctx.current_structure)
        if self.is_U():
            self.out('hubbard', self.ctx.workchains_hp[-1].outputs.hubbard)
        elif self.is_U_V():
            self.out('hubbard_parameters', self.ctx.workchains_hp[-1].outputs.hubbard_parameters)
