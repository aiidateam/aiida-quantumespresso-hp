# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position,wildcard-import
"""Module for the command line interface."""
from aiida.cmdline.groups import VerdiCommandGroup
from aiida.cmdline.params import options, types
import click


@click.group('aiida-hubbard', cls=VerdiCommandGroup, context_settings={'help_option_names': ['-h', '--help']})
@options.PROFILE(type=types.ProfileParamType(load_profile=True))
def cmd_root(profile):  # pylint: disable=unused-argument
    """CLI for the `aiida-hubbard` plugin."""


from .calculations import cmd_calculation
from .workflows import cmd_workflow
