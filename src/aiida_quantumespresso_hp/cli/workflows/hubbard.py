# -*- coding: utf-8 -*-
"""Command line scripts to launch a `SelfConsistentHubbardWorkChain` for testing and demonstration purposes."""
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators
from aiida_quantumespresso.cli.utils import launch
from aiida_quantumespresso.cli.utils import options as options_qe
import click

from . import cmd_launch


@cmd_launch.command('hubbard')
@click.option(
    '--pw',
    'code_pw',
    type=types.CodeParamType(entry_point='quantumespresso.pw'),
    required=True,
    help='The code to use for the pw.x executable.'
)
@click.option(
    '--hp',
    'code_hp',
    type=types.CodeParamType(entry_point='quantumespresso.hp'),
    required=True,
    help='The code to use for the hp.x executable.'
)
@options_qe.STRUCTURE()
@options_qe.PSEUDO_FAMILY()
@options_qe.KPOINTS_MESH(required=True, help='The k-point mesh to use for the SCF calculations.')
@options_qe.QPOINTS_MESH(required=True, help='The q-point mesh to use for the linear response calculation.')
@options_qe.ECUTWFC()
@options_qe.ECUTRHO()
@options_qe.HUBBARD_U(required=True)
@options_qe.STARTING_MAGNETIZATION()
@options_qe.MAX_NUM_MACHINES()
@options_qe.MAX_WALLCLOCK_SECONDS()
@options_qe.WITH_MPI()
@options_qe.DAEMON()
@click.option(
    '--meta-convergence',
    is_flag=True,
    default=False,
    help='Switch on the meta-convergence for the Hubbard parameters.'
)
@click.option(
    '--parallelize-atoms',
    is_flag=True,
    default=False,
    help='Parallelize the linear response calculation over the Hubbard atoms.'
)
@decorators.with_dbenv()
def launch_workflow(
    code_pw, code_hp, structure, pseudo_family, kpoints_mesh, qpoints_mesh, ecutwfc, ecutrho, hubbard_u,
    starting_magnetization, max_num_machines, max_wallclock_seconds, daemon, meta_convergence, parallelize_atoms,
    with_mpi
):
    """Run the `SelfConsistentHubbardWorkChain` for a given input structure."""
    from aiida import orm
    from aiida.plugins import WorkflowFactory
    from aiida_quantumespresso.utils.resources import get_default_options

    cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structure)

    parameters = {
        'SYSTEM': {
            'ecutwfc': ecutwfc or cutoff_wfc,
            'ecutrho': ecutrho or cutoff_rho,
            'lda_plus_u': True,
        },
        'ELECTRONS': {
            'mixing_beta': 0.4,
        }
    }

    parameters_hp = {'INPUTHP': {}}

    structure_kinds = structure.get_kind_names()
    hubbard_u_kinds = [kind for kind, value in hubbard_u]

    if not set(hubbard_u_kinds).issubset(structure_kinds):
        raise click.BadParameter(
            f'the kinds in the specified starting Hubbard U values {hubbard_u_kinds} is not a strict subset of the '
            f'kinds in the structure {structure_kinds}',
            param_hint='hubbard_u'
        )

    if starting_magnetization:

        parameters['SYSTEM']['nspin'] = 2

        for kind, magnetization in starting_magnetization:

            if kind not in structure_kinds:
                raise click.BadParameter(
                    f'the provided structure does not contain the kind {kind}', param_hint='starting_magnetization'
                )

            parameters['SYSTEM'].setdefault('starting_magnetization', {})[kind] = magnetization

    inputs = {
        'structure': structure,
        'hubbard_u': orm.Dict(dict=dict(hubbard_u)),
        'meta_convergence': orm.Bool(meta_convergence),
        'recon': {
            'kpoints': kpoints_mesh,
            'pw': {
                'code': code_pw,
                'pseudos': pseudo_family.get_pseudos(structure=structure),
                'parameters': orm.Dict(dict=parameters),
                'metadata': {
                    'options': get_default_options(max_num_machines, max_wallclock_seconds, with_mpi)
                }
            },
        },
        'relax': {
            'meta_convergence': orm.Bool(False),
            'base': {
                'kpoints': kpoints_mesh,
                'pw': {
                    'code': code_pw,
                    'pseudos': pseudo_family.get_pseudos(structure=structure),
                    'parameters': orm.Dict(dict=parameters),
                    'metadata': {
                        'options': get_default_options(max_num_machines, max_wallclock_seconds, with_mpi)
                    }
                }
            }
        },
        'scf': {
            'kpoints': kpoints_mesh,
            'pw': {
                'code': code_pw,
                'pseudos': pseudo_family.get_pseudos(structure=structure),
                'parameters': orm.Dict(dict=parameters),
                'metadata': {
                    'options': get_default_options(max_num_machines, max_wallclock_seconds, with_mpi)
                }
            }
        },
        'hubbard': {
            'hp': {
                'code': code_hp,
                'qpoints': qpoints_mesh,
                'parameters': orm.Dict(dict=parameters_hp),
                'metadata': {
                    'options': get_default_options(max_num_machines, max_wallclock_seconds, with_mpi)
                }
            },
            'parallelize_atoms': orm.Bool(parallelize_atoms),
        }
    }

    launch.launch_process(WorkflowFactory('quantumespresso.hp.hubbard'), daemon, **inputs)
