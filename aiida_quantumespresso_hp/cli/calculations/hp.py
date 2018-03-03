# -*- coding: utf-8 -*-
import click
from aiida.utils.cli import command
from aiida.utils.cli import options


@command()
@options.code(callback_kwargs={'entry_point': 'quantumespresso.hp'})
@options.calculation(callback_kwargs={'entry_point': 'quantumespresso.pw'})
@options.kpoint_mesh()
@options.max_num_machines()
@options.max_wallclock_seconds()
@options.daemon()
def launch(code, calculation, kpoints, max_num_machines, max_wallclock_seconds, daemon):
    """
    Run a HpCalculation for a previously completed PwCalculation
    """
    from aiida.orm import load_node
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.data.upf import get_pseudos_from_structure
    from aiida.orm.utils import CalculationFactory
    from aiida.work.launch import run_get_pid, submit
    from aiida_quantumespresso.utils.resources import get_default_options
    from aiida_quantumespresso_hp.utils.validation import validate_parent_calculation

    HpCalculation = CalculationFactory('quantumespresso.hp')

    try:
        validate_parent_calculation(calculation)
    except ValueError as exception:
        raise click.BadParameter('invalid parent calculation: {}'.format(exception))

    parameters = {
        'INPUTHP': {
        }
    }

    inputs = {
        'code': code,
        'qpoints': kpoints,
        'parameters': ParameterData(dict=parameters),
        'parent_folder': calculation.out.remote_folder,
        'options': get_default_options(max_num_machines, max_wallclock_seconds),
    }

    click.echo('Running a hp.x calculation ... ')

    process = HpCalculation.process()

    if daemon:
        calculation = submit(process, **inputs)
        pk = calculation.pk
        click.echo('Submitted {}<{}> to the daemon'.format(HpCalculation.__name__, calculation.pk))
    else:
        results, pk = run_get_pid(process, **inputs)

    calculation = load_node(pk)

    click.echo('HpCalculation<{}> terminated with state: {}'.format(pk, calculation.get_state()))
    click.echo('\n{link:25s} {node}'.format(link='Output link', node='Node pk and type'))
    click.echo('{s}'.format(s='-'*60))
    for link, node in sorted(calculation.get_outputs(also_labels=True)):
        click.echo('{:25s} <{}> {}'.format(link, node.pk, node.__class__.__name__))