# -*- coding: utf-8 -*-
"""Command line scripts to launch a `SelfConsistentHubbardWorkChain` for testing and demonstration purposes."""
import click

from aiida.cmdline.params import options
from aiida.cmdline.utils import decorators
from aiida_quantumespresso.cli.utils import options as options_qe


@click.command()
@options.CODE('--pw', 'code_pw', callback_kwargs={'entry_point': 'quantumespresso.pw'})
@options.CODE('--hp', 'code_hp', callback_kwargs={'entry_point': 'quantumespresso.hp'})
@options_qe.STRUCTURE()
@options_qe.PSEUDO_FAMILY()
@options_qe.KPOINTS_MESH('-k', 'kpoints', help='the k-point mesh to use for the SCF calculations')
@options_qe.KPOINTS_MESH('-q', 'qpoints', help='the q-point mesh to use for the linear response calculation')
@options_qe.ECUTWFC()
@options_qe.ECUTRHO()
@options_qe.HUBBARD_U()
@options_qe.STARTING_MAGNETIZATION()
@options_qe.AUTOMATIC_PARALLELIZATION()
@options_qe.CLEAN_WORKDIR()
@options_qe.MAX_NUM_MACHINES()
@options_qe.MAX_WALLCLOCK_SECONDS()
@options_qe.DAEMON()
@click.option(
    '--meta-convergence',
    is_flag=True,
    default=False,
    show_default=True,
    help='switch on the meta-convergence for the Hubbard parameters'
)
@click.option(
    '--is-insulator', is_flag=True, default=False, show_default=True, help='treat the structure as an insulator'
)
@click.option(
    '--parallelize-atoms',
    is_flag=True,
    default=False,
    show_default=True,
    help='parallelize the linear response calculation over the Hubbard atoms'
)
@decorators.with_dbenv()
def launch(
    code_pw, code_hp, structure, pseudo_family, kpoints, qpoints, ecutwfc, ecutrho, hubbard_u, starting_magnetization,
    automatic_parallelization, clean_workdir, max_num_machines, max_wallclock_seconds, daemon, meta_convergence,
    is_insulator, parallelize_atoms
):
    """Run the `SelfConsistentHubbardWorkChain` for a given input structure."""
    from aiida import orm
    from aiida.engine import run, submit
    from aiida.plugins import WorkflowFactory
    from aiida_quantumespresso.utils.resources import get_default_options

    SelfConsistentHubbardWorkChain = WorkflowFactory('quantumespresso.hp.hubbard')

    parameters = {
        'SYSTEM': {
            'ecutwfc': ecutwfc,
            'ecutrho': ecutrho,
            'lda_plus_u': True,
        },
    }

    parameters_hp = {'INPUTHP': {}}

    structure_kinds = structure.get_kind_names()
    hubbard_u_kinds = [kind for kind, value in hubbard_u]

    if not set(hubbard_u_kinds).issubset(structure_kinds):
        raise click.BadParameter(
            'the kinds in the specified starting Hubbard U values {} is not a strict subset of the kinds in the '
            'structure {}'.format(hubbard_u_kinds, structure_kinds),
            param_hint='hubbard_u'
        )

    if starting_magnetization:

        parameters['SYSTEM']['nspin'] = 2

        for kind, magnetization in starting_magnetization:

            if kind not in structure_kinds:
                raise click.BadParameter(
                    'the provided structure does not contain the kind {}'.format(kind),
                    param_hint='starting_magnetization'
                )

            parameters['SYSTEM'].setdefault('starting_magnetization', {})[kind] = magnetization

    inputs = {
        'structure': structure,
        'hubbard_u': orm.Dict(dict=hubbard_u),
        'meta_convergence': orm.Bool(meta_convergence),
        'is_insulator': orm.Bool(is_insulator),
        'scf': {
            'code': code_pw,
            'pseudo_family': orm.Str(pseudo_family),
            'kpoints': kpoints,
            'parameters': orm.Dict(dict=parameters),
            'options': orm.Dict(dict=get_default_options(max_num_machines, max_wallclock_seconds))
        },
        'hp': {
            'code': code_hp,
            'qpoints': qpoints,
            'parameters': orm.Dict(dict=parameters_hp),
            'options': orm.Dict(dict=get_default_options(max_num_machines, max_wallclock_seconds)),
            'parallelize_atoms': orm.Bool(parallelize_atoms),
        }
    }

    if daemon:
        workchain = submit(SelfConsistentHubbardWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(SelfConsistentHubbardWorkChain.__name__, workchain.pk))
    else:
        run(SelfConsistentHubbardWorkChain, **inputs)
