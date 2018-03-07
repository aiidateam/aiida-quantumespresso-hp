# -*- coding: utf-8 -*-
import click
from aiida.utils.cli import command
from aiida.utils.cli import options
from aiida_quantumespresso.utils.cli import options as options_qe


@command()
@options.code(callback_kwargs={'entry_point': 'quantumespresso.hp'})
@options.calculation(callback_kwargs={'entry_point': 'quantumespresso.pw'})
@options.kpoint_mesh()
@options.max_num_machines()
@options.max_wallclock_seconds()
@options.daemon()
@options_qe.clean_workdir()
def launch(
    code, calculation, kpoints, max_num_machines, max_wallclock_seconds, daemon, clean_workdir):
    """
    Run the HpWorkChain for a completed Hubbard PwCalculation
    """
    from aiida.orm.data.base import Bool
    from aiida.orm.data.parameter import ParameterData
    from aiida.orm.utils import WorkflowFactory
    from aiida.work.launch import run, submit
    from aiida_quantumespresso.utils.resources import get_default_options

    HpBaseWorkChain = WorkflowFactory('quantumespresso.hp.base')

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
        'options': ParameterData(dict=options)
    }

    if daemon:
        workchain = submit(HpBaseWorkChain, **inputs)
        click.echo('Submitted {}<{}> to the daemon'.format(HpBaseWorkChain.__name__, workchain.pk))
    else:
        run(HpBaseWorkChain, **inputs)