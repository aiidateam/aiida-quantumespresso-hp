# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Tests for the `collect_atomic_calculations` calculation function."""

import contextlib
import io
import os
import shutil
import tempfile

from aiida import orm

from aiida_quantumespresso_hp.calculations.hp import HpCalculation
from aiida_quantumespresso_hp.calculations.functions.collect_atomic_calculations import collect_atomic_calculations


@contextlib.contextmanager
def tempdir():
    """Return the absolute filepath of a temporary directory."""
    dirpath = tempfile.mkdtemp()
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


def create_file_tree(directory, tree):
    """Create a file tree in the given directory.

    :param directory: the absolute path of the directory into which to create the tree
    :param tree: a dictionary representing the tree structure
    """
    for key, value in tree.items():
        if isinstance(value, dict):
            subdir = os.path.join(directory, key)
            os.makedirs(subdir)
            create_file_tree(subdir, value)
        else:
            with io.open(os.path.join(directory, key), 'w', encoding='utf8') as handle:
                handle.write(value)


def create_retrieved_folder(tree):
    """Return a `FolderData` with the contents of `tree` copied within it."""
    with tempdir() as dirpath:
        create_file_tree(dirpath, tree)
        retrieved = orm.FolderData()
        retrieved.put_object_from_tree(dirpath)

    return retrieved


def test_collect_atomic_calculations(fixture_database, fixture_computer_localhost):
    """Test the `collect_atomic_calculations` calculation function."""
    number_perturbations = 2

    retrieved_folders = {}

    for index in range(number_perturbations):
        tree = {
            'out': {
                'aiida.save': {
                    'data-file.xml': u'',
                },
                'aiida.occup': u'',
                'aiida.paw': u'',
                'HP': {
                    'aiida.chi.pert_{}.dat'.format(index): u'',
                }
            }
        }
        retrieved_folders['site_index_{}'.format(index)] = create_retrieved_folder(tree)

    result = collect_atomic_calculations(**retrieved_folders)

    assert isinstance(result, orm.FolderData)
    assert len(result.list_object_names(HpCalculation.dirname_output_hubbard)) == number_perturbations
