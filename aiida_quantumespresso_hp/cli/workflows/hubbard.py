# -*- coding: utf-8 -*-
import click
from aiida.utils.cli import command
from aiida.utils.cli import options
from aiida_quantumespresso.utils.cli import options as options_qe


@command()
@options.code('--pw', 'code_pw', callback_kwargs={'entry_point': 'quantumespresso.pw'})
@options.code('--hp', 'code_hp', callback_kwargs={'entry_point': 'quantumespresso.hp'})
@click.option('-h', '--hubbard-kind', multiple=True, nargs=2, type=click.Tuple([unicode, float]))
@options.structure()
@options.pseudo_family()
@options.kpoint_mesh()
@options.max_num_machines()
@options.max_wallclock_seconds()
@options.daemon()
@options_qe.automatic_parallelization()
@options_qe.clean_workdir()
def launch(
    code_pw, code_hp, hubbard_kind, structure, pseudo_family, kpoints, max_num_machines, max_wallclock_seconds, daemon,
    automatic_parallelization, clean_workdir):
    """
    Run the SelfConsistentHubbardWorkChain for a given input structure
    """
    from aiida.orm.data.base import Bool, Str
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.utils import WorkflowFactory
    from aiida.work.launch import run, submit
    from aiida_quantumespresso.utils.resources import get_default_options

    SelfConsistentHubbardWorkChain = WorkflowFactory('quantumespresso.hp.hubbard')

    parameters = {
        'SYSTEM': {
            'ecutwfc': 30.,
            'ecutrho': 240.,
            'lda_plus_u': True,
        },
    }

    parameters_hp = {
        'INPUTHP': {
        }
    }

    options = get_default_options(max_num_machines, max_wallclock_seconds)

    hubbard_u = {
        'Co': 1E-8
    }

    settings = {}

    inputs = {
        'structure': structure,
        'hubbard_u': ParameterData(dict=hubbard_u),
        'is_insulator': Bool(True),
        'scf': {
            'code': code_pw,
            'pseudo_family': Str(pseudo_family),
            'kpoints': kpoints,
            'parameters': ParameterData(dict=parameters),
            'options': ParameterData(dict=options)
        },
        'relax': {
            'code': code_pw,
            'pseudo_family': Str(pseudo_family),
            'kpoints': kpoints,
            'parameters': ParameterData(dict=parameters),
            'options': ParameterData(dict=options),
            'meta_convergence': Bool(False),
        },
        'hp': {
            'code': code_hp,
            'qpoints': kpoints,
            'parameters': ParameterData(dict=parameters_hp),
            'settings': ParameterData(dict=settings),
            'options': ParameterData(dict=options),
        }
    }

    if daemon:
        workchain = submit(SelfConsistentHubbardWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(SelfConsistentHubbardWorkChain.__name__, workchain.pk))
    else:
        run(SelfConsistentHubbardWorkChain, **inputs)