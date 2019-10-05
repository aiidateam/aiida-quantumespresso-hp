# -*- coding: utf-8 -*-
# pylint: disable=cyclic-import,reimported,unused-import,wrong-import-position
"""Module with CLI commands for the various calculation job implementations."""
from .. import cmd_root


@cmd_root.group('calculation')
def cmd_calculation():
    """Commands to launch and interact with calculations."""


@cmd_calculation.group('launch')
def cmd_launch():
    """Launch calculations."""


# Import the sub commands to register them with the CLI
from .hp import launch_calculation
