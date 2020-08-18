# -*- coding: utf-8 -*-
"""Define the setup for the `aiida-quantumespresso-hp` plugin."""


def setup_package():
    """Install the `aiida-quantumespresso-hp` package."""
    import json
    from setuptools import setup, find_packages
    try:
        import fastentrypoints  # pylint: disable=unused-import
    except ImportError:
        # This should only occur when building the package, i.e. when
        # executing 'python setup.py sdist' or 'python setup.py bdist_wheel'
        pass

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
