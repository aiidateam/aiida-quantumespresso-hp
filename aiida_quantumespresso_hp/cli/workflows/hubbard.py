# -*- coding: utf-8 -*-
import click

from aiida.cmdline.params import options
from aiida.cmdline.utils import decorators
from aiida_quantumespresso.utils.cli import options as options_qe


@click.command()
@options.code('--pw', 'code_pw', callback_kwargs={'entry_point': 'quantumespresso.pw'})
@options.code('--hp', 'code_hp', callback_kwargs={'entry_point': 'quantumespresso.hp'})
@options.structure()
@options.pseudo_family()
@options.kpoint_mesh('-k', 'kpoints', help='the k-point mesh to use for the SCF calculations')
@options.kpoint_mesh('-q', 'qpoints', help='the q-point mesh to use for the linear response calculation')
@options_qe.ecutwfc()
@options_qe.ecutrho()
@options_qe.hubbard_u()
@options_qe.starting_magnetization()
@options_qe.automatic_parallelization()
@options_qe.clean_workdir()
@options.max_num_machines()
@options.max_wallclock_seconds()
@options.daemon()
@click.option(
    '--meta-convergence', is_flag=True, default=False, show_default=True,
    help='switch on the meta-convergence for the Hubbard parameters'
)
@click.option(
    '--is-insulator', is_flag=True, default=False, show_default=True,
    help='treat the structure as an insulator'
)
@click.option(
    '--parallelize-atoms', is_flag=True, default=False, show_default=True,
    help='parallelize the linear response calculation over the Hubbard atoms'
)
@decorators.with_dbenv()
def launch(
    code_pw, code_hp, structure, pseudo_family, kpoints, qpoints, ecutwfc, ecutrho, hubbard_u, starting_magnetization,
    automatic_parallelization, clean_workdir, max_num_machines, max_wallclock_seconds, daemon, meta_convergence,
    is_insulator, parallelize_atoms):
    """
    Run the SelfConsistentHubbardWorkChain for a given input structure
    """
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

    parameters_hp = {
        'INPUTHP': {
        }
    }

    options = get_default_options(max_num_machines, max_wallclock_seconds)

    structure_kinds = structure.get_kind_names()
    hubbard_u_kinds = [kind for kind, value in hubbard_u]
    hubbard_u = {kind: value for kind, value in hubbard_u}

    if not set(hubbard_u_kinds).issubset(structure_kinds):
        raise click.BadParameter(
            'the kinds in the specified starting Hubbard U values {} is not a strict subset of the kinds in the structure {}'.format(
            hubbard_u_kinds, structure_kinds), param_hint='hubbard_u'
        )

    if starting_magnetization:

        parameters['SYSTEM']['nspin'] = 2

        for kind, magnetization in starting_magnetization:

            if kind not in structure_kinds:
                raise click.BadParameter('the provided structure does not contain the kind {}'.format(kind), param_hint='starting_magnetization')

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
            'options': orm.Dict(dict=options)
        },
        'hp': {
            'code': code_hp,
            'qpoints': qpoints,
            'parameters': orm.Dict(dict=parameters_hp),
            'options': orm.Dict(dict=options),
            'parallelize_atoms': orm.Bool(parallelize_atoms),
        }
    }

    if daemon:
        workchain = submit(SelfConsistentHubbardWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(SelfConsistentHubbardWorkChain.__name__, workchain.pk))
    else:
        run(SelfConsistentHubbardWorkChain, **inputs)
