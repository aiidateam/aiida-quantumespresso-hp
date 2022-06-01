# -*- coding: utf-8 -*-
"""Command line scripts to launch a `HpWorkChain` for testing and demonstration purposes."""

from aiida.cmdline.params import options, types
from aiida.cmdline.utils import decorators
from aiida_quantumespresso.cli.utils import launch
from aiida_quantumespresso.cli.utils import options as options_qe
import click

from .. import cmd_launch


@cmd_launch.command('hp-main')
@options.CODE(required=True, type=types.CodeParamType(entry_point='quantumespresso.hp'))
@options_qe.KPOINTS_MESH(default=[1, 1, 1])
@options_qe.PARENT_FOLDER()
@options_qe.MAX_NUM_MACHINES()
@options_qe.MAX_WALLCLOCK_SECONDS()
@options_qe.WITH_MPI()
@options_qe.DAEMON()
@options.DRY_RUN()
@options_qe.CLEAN_WORKDIR()
@click.option(
    '--parallelize-atoms',
    is_flag=True,
    default=False,
    show_default=True,
    help='Parallelize the linear response calculation over the Hubbard atoms.'
)
@decorators.with_dbenv()
def launch_workflow(
    code, kpoints_mesh, parent_folder, max_num_machines, max_wallclock_seconds, with_mpi, clean_workdir,
    parallelize_atoms, daemon, dry_run
):
    """Run a `HpWorkChain`."""
    from aiida import orm
    from aiida.plugins import WorkflowFactory
    from aiida_quantumespresso.utils.resources import get_default_options

    parameters = {'INPUTHP': {}}

    inputs = {
        'hp': {
            'code': code,
            'qpoints': kpoints_mesh,
            'parameters': orm.Dict(dict=parameters),
            'parent_scf': parent_folder,
            'metadata': {
                'options': get_default_options(max_num_machines, max_wallclock_seconds, with_mpi),
            },
        },
        'parallelize_atoms': orm.Bool(parallelize_atoms),
        'clean_workdir': orm.Bool(clean_workdir),
    }

    if dry_run:
        if daemon:
            raise click.BadParameter('cannot send to the daemon if in dry_run mode', param_hint='--daemon')
        inputs.setdefault('metadata', {})['store_provenance'] = False
        inputs['metadata']['dry_run'] = True

    launch.launch_process(WorkflowFactory('quantumespresso.hp.main'), daemon, **inputs)
