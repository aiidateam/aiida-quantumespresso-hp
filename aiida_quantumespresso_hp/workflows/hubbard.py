# -*- coding: utf-8 -*-
from copy import deepcopy
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code
from aiida.orm.data.base import Bool, Float, Int, Str
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.utils import CalculationFactory, WorkflowFactory
from aiida.orm.data.array.bands import find_bandgap
from aiida.work.run import submit
from aiida.work.workchain import WorkChain, ToContext, while_, if_, append_
from aiida_quantumespresso.utils.defaults.calculation import pw as qe_defaults

PwCalculation = CalculationFactory('quantumespresso.pw')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
HpWorkChain = WorkflowFactory('quantumespresso.hp.main')

class SelfConsistentHubbardWorkChain(WorkChain):
    """
    Workchain that for a given input structure will compute the self-consistent Hubbard U parameters
    by iteratively relaxing the structure with the PwRelaxWorkChain and computing the Hubbard U
    parameters through the HpWorkChain, until the Hubbard U values are converged within a certain tolerance.

    The procedure in each step of the convergence cycle is slightly different depending on the electronic and
    magnetic properties of the system. Each cycle will roughly consist of three steps:

        * Relaxing the structure at the current Hubbard U values
        * One or more SCF calculations depending on the system's electronic and magnetic properties
        * A self-consistent calculation of the Hubbard U parameters, restarted from the previous SCF run

    The possible options for the set of SCF calculations that have to be run in the second step look are:

        * Metals:
            - SCF with smearing

        * Non-magnetic insulators
            - SCF with fixed occupations

        * Magnetic insulators
            - SCF with smearing
            - SCF with fixed occupations, where total magnetization and number of bands are fixed
              to the values found from the previous SCF calculation

    When convergence is achieved a ParameterData node will be returned containing the final converged
    Hubbard U parameters.
    """

    def __init__(self, *args, **kwargs):
        super(SelfConsistentHubbardWorkChain, self).__init__(*args, **kwargs)

        # Default values
        self.defaults = AttributeDict({
            'qe': qe_defaults,
        })

    @classmethod
    def define(cls, spec):
        super(SelfConsistentHubbardWorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('hubbard_u', valid_type=ParameterData)
        spec.input('tolerance', valid_type=Float, default=Float(0.1))
        spec.input('max_iterations', valid_type=Int, default=Int(5))
        spec.input('is_insulator', valid_type=Bool, required=False)
        spec.input_group('scf', required=False)
        spec.input_group('relax', required=False)
        spec.input_group('hp')
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            if_(cls.should_run_recon)(
                cls.run_recon,
                cls.inspect_recon,
            ),
            while_(cls.should_run_iteration)(
                if_(cls.should_run_relax)(
                    cls.run_relax,
                    cls.inspect_relax,
                ),
                if_(cls.is_metal)(
                    cls.run_scf_smearing
                ).elif_(cls.is_magnetic)(
                    cls.run_scf_smearing,
                    cls.run_scf_fixed_magnetic
                ).else_(
                    cls.run_scf_fixed
                ),
                cls.run_hp,
                cls.inspect_hp,
            ),
            cls.run_results,
        )

    def setup(self):
        """
        Input validation and context setup
        """
        self.ctx.current_structure = self.inputs.structure
        self.ctx.current_hubbard_u = self.inputs.hubbard_u.get_dict()
        self.ctx.max_iterations = self.inputs.max_iterations.value
        self.ctx.is_converged = False
        self.ctx.is_magnetic = None
        self.ctx.is_metal = None
        self.ctx.iteration = 1
        self.ctx.skip_relax = None

        structure_kinds = self.inputs.structure.get_kind_names()
        hubbard_u_kinds = self.inputs.hubbard_u.get_dict().keys()

        if not set(hubbard_u_kinds).issubset(structure_kinds):
            self.abort_nowait('the kinds in the specified starting Hubbard U values is not a strict subset of the kinds in the structure')
            return

        if 'relax' not in self.inputs and 'scf' not in self.inputs:
            self.abort_nowait("neither the 'relax' nor 'scf' inputs have been defined")
            return

        if 'relax' in self.inputs:
            self.ctx.skip_relax = False
            input_group = self.inputs.relax
        else:
            self.ctx.skip_relax = True
            input_group = self.inputs.scf

        self.ctx.inputs_raw = AttributeDict(input_group)

        # If provided in the input_group inputs, unwrap the parameters ParameterData node, else specify empty dict
        if 'parameters' in self.ctx.inputs_raw:
            self.ctx.inputs_raw.parameters = self.ctx.inputs_raw.parameters.get_dict()
        else:
            self.ctx.inputs_raw.parameters = {}

        # Ensure the CONTROL, SYSTEM and ELECTRONS cards are defined in the parameters input node
        self.ctx.inputs_raw.parameters.setdefault('CONTROL', {})
        self.ctx.inputs_raw.parameters.setdefault('SYSTEM', {})
        self.ctx.inputs_raw.parameters.setdefault('ELECTRONS', {})

        # Set the current structure and current Hubbard U values to the input values
        self.ctx.inputs_raw.structure = self.ctx.current_structure
        self.ctx.inputs_raw.parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u

        # Determine whether the system is to be treated as magnetic
        system = self.ctx.inputs_raw.parameters['SYSTEM']
        if 'nspin' in system and system.get('nspin', self.defaults.qe.nspin) != 1:
            self.report('system is determined to be magnetic')
            self.ctx.is_magnetic = True
        else:
            self.report('system is determined to be non-magnetic')
            self.ctx.is_magnetic = False

        if 'is_insulator' in self.inputs:
            self.ctx.is_metal = not self.inputs.is_insulator

    def validate_inputs(self):
        """
        Validate inputs that may depend on each other
        """
        pass

    def should_run_recon(self):
        """
        Returns whether a reconnaissance calculation needs to be run, which is the case if it is yet unknown
        whether the system is metallic or insulating
        """
        return self.ctx.is_metal is None

    def should_run_relax(self):
        """
        Returns whether the structure should be relaxed before every Hubbard iteration, If the input 'relax'
        is not defined, the relax part is skipped in each iteration and the input structure of the workchain
        is used throughout the workchain
        """
        return not self.ctx.skip_relax

    def run_recon(self):
        """
        Run a PwBaseWorkChain to compute a simple scf cycle with one step with smearing turned on to determine
        whether the system is most likely a metal or an insulator. This step is required because the metallicity
        of the systems determines how the relaxation calculations in the convergence cycle have to be performed.
        """
        inputs = deepcopy(self.ctx.inputs_raw)

        inputs.parameters['CONTROL']['calculation'] = 'scf'
        inputs.parameters['ELECTRONS']['scf_must_converge'] = False
        inputs.parameters['ELECTRONS']['electron_maxstep'] = 1
        inputs.parameters['SYSTEM']['occupations'] = 'smearing'
        inputs.parameters['SYSTEM']['smearing'] = 'marzari-vanderbilt'
        inputs.parameters['SYSTEM']['degauss'] = 0.001

        inputs.update({
            'parameters': ParameterData(dict=inputs.parameters),
        })

        running = submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> reconnaissance'.format(running.pid))

        return ToContext(workchain_recon=running)

    def inspect_recon(self):
        """
        Analyze the result of the reconnaissance run to determine whether the system is an insulator or a metal
        """
        try:
            workchain = self.ctx.workchain_recon
        except AttributeError:
            self.abort_nowait('the run_recon step did not return a PwBaseWorkChain')
            return

        bands = workchain.out.output_band
        parameters = workchain.out.output_parameters.get_dict()
        number_electrons = parameters['number_of_electrons']

        is_insulator, gap = find_bandgap(bands, number_electrons=number_electrons)

        if is_insulator:
            self.report('system is determined to be an insulator')
            self.ctx.is_metal = False
        else:
            self.report('system is determined to be a metal')
            self.ctx.is_metal = True

    def should_run_iteration(self):
        """
        Return whether another iteration of the self-consistent cycle should be run, which is the case as long as
        the Hubbard parameters are not yet converged and the maximum number of iterations has not been exceeded
        """
        return not self.ctx.is_converged and self.ctx.iteration < self.ctx.max_iterations

    def is_metal(self):
        """
        Return whether the current structure is a metal 
        """
        return self.ctx.is_metal

    def is_magnetic(self):
        """
        Return whether the current structure is magnetic
        """
        return self.ctx.is_magnetic

    def run_relax(self):
        """
        Run the PwRelaxWorkChain to run a relax PwCalculation
        """
        inputs = deepcopy(self.ctx.inputs_raw)

        inputs.update({
            'parameters': ParameterData(dict=inputs.parameters)
        })

        running = submit(PwRelaxWorkChain, **inputs)

        self.report('launching PwRelaxWorkChain<{}> iteration #{}'.format(running.pid, self.ctx.iteration))

        return ToContext(workchains_relax=append_(running))

    def inspect_relax(self):
        """
        Analyze the result of the PwRelaxWorkChain by verifying that an output structure was returned and if
        so setting the current structure and parameters in the context to the values returned by the workchain
        """
        try:
            workchain = self.ctx.workchains_relax[-1]
        except IndexError:
            self.abort_nowait('the first iteration finished without returning a PwRelaxWorkChain')
            return

        try:
            structure = workchain.out.output_structure
        except AttributeError as exception:
            self.abort_nowait('the relax workchain did not have an output structure and probably failed')
            return

        self.ctx.current_structure = structure
        self.ctx.current_parameters = workchain.out.output_parameters

        # Self update the current structure in the inputs
        self.ctx.inputs_raw.structure = structure

    def run_scf_fixed(self):
        """
        Run a simple PwCalculation with fixed occupations
        """
        inputs = deepcopy(self.ctx.inputs_raw)

        inputs.parameters['CONTROL']['calculation'] = 'scf'
        inputs.parameters['SYSTEM']['occupations'] = 'fixed'
        inputs.parameters['SYSTEM'].pop('degauss', None)
        inputs.parameters['SYSTEM'].pop('smearing', None)

        inputs.update({
            'parameters': ParameterData(dict=inputs.parameters)
        })

        running = submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> with fixed occupations'.format(running.pid))

        return ToContext(workchains_scf=append_(running))

    def run_scf_smearing(self):
        """
        Run a simple PwCalculation with smeared occupations
        """
        inputs = deepcopy(self.ctx.inputs_raw)

        inputs.parameters['CONTROL']['calculation'] = 'scf'
        inputs.parameters['SYSTEM']['occupations'] = 'smearing'
        inputs.parameters['SYSTEM']['smearing'] = 'marzari-vanderbilt'
        inputs.parameters['SYSTEM']['degauss'] = 0.001

        inputs.update({
            'parameters': ParameterData(dict=inputs.parameters)
        })

        running = submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> with smeared occupations'.format(running.pid))

        return ToContext(workchains_scf=append_(running))

    def run_scf_fixed_magnetic(self):
        """
        Run a simple PwCalculation with fixed occupations restarting from the previous calculation
        with smeared occupations, based on which the number of bands and total magnetization are fixed
        """
        inputs = deepcopy(self.ctx.inputs_raw)

        previous_workchain = self.ctx.workchains_scf[-1]
        previous_parameters = previous_workchain.out.output_parameters

        inputs.parameters['CONTROL']['calculation'] = 'scf'
        inputs.parameters['SYSTEM']['nbnd'] = previous_parameters.get_dict()['number_of_bands']
        inputs.parameters['SYSTEM']['total_magnetization'] = previous_parameters.get_dict()['total_magnetization']
        inputs.parameters['SYSTEM']['occupations'] = 'fixed'
        inputs.parameters['SYSTEM'].pop('degauss', None)
        inputs.parameters['SYSTEM'].pop('smearing', None)

        inputs.update({
            'parameters': ParameterData(dict=inputs.parameters)
        })

        running = submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> with fixed occupations, bands and total magnetization'.format(running.pid))

        return ToContext(workchains_scf=append_(running))

    def run_hp(self):
        """
        Run the HpWorkChain restarting from the last completed scf calculation
        """
        workchain = self.ctx.workchains_scf[-1]
        parent_calculation = workchain.out.output_parameters.get_inputs(node_type=PwCalculation)[0]

        inputs = self.inputs.hp
        inputs.update({
            'parent_calculation': parent_calculation
        })

        running = submit(HpWorkChain, **inputs)

        self.report('launching HpWorkChain<{}> iteration #{}'.format(running.pid, self.ctx.iteration))

        return ToContext(workchains_hp=append_(running))

    def inspect_hp(self):
        """
        Analyze the last completed HpWorkChain. We check the current Hubbard U parameters and compare those with
        the values computed in the previous iteration. If the difference for all Hubbard sites is smaller than
        the tolerance, the calculation is considered to be converged.
        """
        try:
            workchain = self.ctx.workchains_hp[-1]
        except IndexError:
            self.abort_nowait('the first iteration finished without returning a HpWorkChain')
            return

        try:
            hubbard = workchain.out.output_hubbard
        except AttributeError as exception:
            self.abort_nowait('the Hp workchain did not have a output_hubbard node and probably failed')
            return

        prev_hubbard_u = self.ctx.current_hubbard_u
        curr_hubbard_u = {}

        for entry in hubbard.get_dict()['sites']:
            curr_hubbard_u[entry['kind']] = float(entry['value'])

        # Compare new Hubbard U with values from previous iteration to check the convergence
        converged = True
        for kind in curr_hubbard_u.keys():
            prev_value = prev_hubbard_u[kind]
            curr_value = curr_hubbard_u[kind]
            if abs(curr_value - prev_value) > self.inputs.tolerance.value:
                converged = False

        self.ctx.current_hubbard_u = curr_hubbard_u
        self.ctx.is_converged = converged

        if converged:
            self.report('Hubbard U parameters are converged')
        else:
            self.report('Hubbard U parameters are not converged')

        self.report('values from previous iteration: {}'.format(' '.join([str(v) for v in prev_hubbard_u.values()])))
        self.report('values from current iteration: {}'.format(' '.join([str(v) for v in curr_hubbard_u.values()])))

        # Set the new Hubbard U values in the input parameters for the next iteration
        self.ctx.inputs_raw.parameters['SYSTEM']['hubbard_u'] = self.ctx.current_hubbard_u
        self.ctx.iteration += 1

        return

    def run_results(self):
        """
        Attach the final converged Hubbard U parameters and the corresponding structure
        """
        self.report('Hubbard U parameters self-consistently converged in {} iterations'.format(self.ctx.iteration))
        self.out('structure', self.ctx.current_structure)
        self.out('hubbard_u', ParameterData(dict=self.ctx.current_hubbard_u))