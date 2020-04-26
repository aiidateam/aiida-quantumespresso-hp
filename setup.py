# -*- coding: utf-8 -*-
"""Define the setup for the `aiida-quantumespresso-hp` plugin."""


def setup_package():
    """Install the `aiida-quantumespresso-hp` package."""
    import json
    from utils import fastentrypoints  # pylint: disable=unused-import
    from setuptools import setup, find_packages

    filename_setup_json = 'setup.json'
    filename_description = 'README.md'

    with open(filename_setup_json, 'r') as handle:
        setup_json = json.load(handle)

    with open(filename_description, 'r') as handle:
        description = handle.read()

    setup(
        include_package_data=True,
        packages=find_packages(),
        long_description=description,
        long_description_content_type='text/markdown',
        **setup_json
    )


if __name__ == '__main__':
    setup_package()
