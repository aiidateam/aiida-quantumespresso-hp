HUBBARD TODO:
=============================

SelfConsistentHubbardWorkChain:
v Make the relax optional, i.e. skip the PwRelax altogether and go straight to scf depending on metallicity/magnetism
v Make 'relax' inputs optional when run_relax is True
v For smearing, only set m-v as default if not explicitly set by the user
v For magnetic/insulator, first scf_smearing run econv = 10^-10, then restart scf_fixed_magnetic 10^-15
v In run_scf_fixed_magnetic set the 'restart' flag in the control parameters
v Set default conv_thr 10^-15, except for run_scf_smearing in MI, except if user defines one, then respect that everywhere
v During each cycle make sure Hubbard U parameters are updated
v Only use the original inputs and make changes to local deep copies
v Use U_projection_type = 'ortho-atomic' for scf runs, just 'atomic' for relax, again respecting value if preset by user, except for when user specifies 'ortho-atomic', then change to 'atomic' for relax, because pw.x will crash, fire WARNING report

PwCalculation:
- Parse electronic and ionic dipole moment for PwCalculation

HpCalculation:
- Check that the PwCalculation was run with lda_plus_U = True, or at the very least deal with that error in the parser

General:
- Phase diagrams with CPLAP
- Default data nodes for workfunctions



launch_calculation_pw -c pw-v5.1 -p SSSP_v0.7_eff_PBE -s 136
launch_calculation_hp -c hp-v5.1 -C 2836
HpCalculation<2844> terminated with state: FAILED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Error in routine d_matrix (3):
     D_S (l=3) for this symmetry operation is not orthogonal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



HUBBARD PARAMETERS ERRORS:
* file occup not found       The calculation from which was restarted did not have any Hubbard sites