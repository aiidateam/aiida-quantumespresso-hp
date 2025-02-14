# -*- coding: utf-8 -*-
"""Command line scripts to launch a `HpCalculation` for testing and demonstration purposes."""
from aiida.cmdline.params import options, types
from aiida.cmdline.utils import decorators
from aiida_quantumespresso.cli.utils import launch
from aiida_quantumespresso.cli.utils import options as options_qe
import click

from . import cmd_launch


@cmd_launch.command('hp')
@options.CODE(required=True, type=types.CodeParamType(entry_point='quantumespresso.hp'))
@options_qe.KPOINTS_MESH(default=[1, 1, 1])
@options_qe.PARENT_FOLDER(type=types.DataParamType(sub_classes=('aiida.data:core.remote',)))
@options_qe.MAX_NUM_MACHINES()
@options_qe.MAX_WALLCLOCK_SECONDS()
@options_qe.WITH_MPI()
@options_qe.DAEMON()
@options.DRY_RUN()
@decorators.with_dbenv()
def launch_calculation(
    code, kpoints_mesh, parent_folder, max_num_machines, max_wallclock_seconds, with_mpi, daemon, dry_run
):
    """Run a `HpCalculation`."""
    from aiida.orm import Dict
    from aiida.plugins import CalculationFactory
    from aiida_quantumespresso.utils.resources import get_default_options

    parameters = {'INPUTHP': {}}

    inputs = {
        'code': code,
        'qpoints': kpoints_mesh,
        'parameters': Dict(parameters),
        'parent_scf': parent_folder,
        'metadata': {
            'options': get_default_options(max_num_machines, max_wallclock_seconds, with_mpi),
        }
    }

    if dry_run:
        if daemon:
            raise click.BadParameter('cannot send to the daemon if in dry_run mode', param_hint='--daemon')
        inputs.setdefault('metadata', {})['store_provenance'] = False
        inputs['metadata']['dry_run'] = True

    launch.launch_process(CalculationFactory('quantumespresso.hp'), daemon, **inputs)
