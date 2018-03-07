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
@click.option('-p', '--parallelize-atoms', is_flag=True, default=False)
def launch(
    code, calculation, kpoints, max_num_machines, max_wallclock_seconds, daemon, parallelize_atoms):
    """
    Run the HpWorkChain for a completed Hubbard PwCalculation
    """
    from aiida.orm.data.base import Bool
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.utils import WorkflowFactory
    from aiida.work.launch import run, submit
    from aiida_quantumespresso.utils.resources import get_default_options

    HpWorkChain = WorkflowFactory('quantumespresso.hp.main')

    parameters = {
        'INPUTHP': {
        }
    }

    options = get_default_options(max_num_machines, max_wallclock_seconds)

    inputs = {
        'code': code,
        'parent_calculation': calculation,
        'qpoints': kpoints,
        'parameters': ParameterData(dict=parameters),
        'settings': ParameterData(dict={}),
        'options': ParameterData(dict=options)
    }

    if parallelize_atoms:
        inputs['parallelize_atoms'] = Bool(True)

    if daemon:
        workchain = submit(HpWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(HpWorkChain.__name__, workchain.pk))
    else:
        run(HpWorkChain, **inputs)