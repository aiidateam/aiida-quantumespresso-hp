# `aiida-hubbard`
AiiDA plugin for the Hubbard module of Quantum ESPRESSO.
The plugin requires HP v7.2 or above and is not compatible with older versions.

This is the official AiiDA plugin for the [HP](https://www.sciencedirect.com/science/article/pii/S0010465522001746) code of [Quantum ESPRESSO](https://www.quantum-espresso.org/).

## Compatibility matrix

The matrix below assumes the user always install the latest patch release of the specified minor version, which is recommended.

| Plugin | AiiDA | Python | Quantum ESPRESSO |
|-|-|-|-|
| `v0.1.0` | ![Compatibility for v4.0][AiiDA v4.0-pydantic2] |  [![PyPI pyversions][Python v3.9-v3.12]](https://pypi.org/project/aiida-quantumespresso/) | ![Quantum ESPRESSO compatibility][QE v7.2-7.4] |

## Installation
To install using pip, simply execute:

    pip install git+https://github.com/aiidateam/aiida-hubbard

or when installing from source:

    git clone https://github.com/aiidateam/aiida-hubbard
    pip install aiida-hubbard

## Pseudopotentials
Pseudopotentials are installed and managed through the [`aiida-pseudo` plugin](https://pypi.org/project/aiida-pseudo/).
The easiest way to install pseudopotentials, is to install a version of the [SSSP](https://www.materialscloud.org/discover/sssp/table/efficiency) through the CLI of `aiida-pseudo`.
Simply run

    aiida-pseudo install sssp

to install the default SSSP version.
List the installed pseudopotential families with the command `aiida-pseudo list`.
You can then use the name of any family in the command line using the `-F` flag.

## Development

### Running tests
To run the tests, simply clone and install the package locally with the [tests] optional dependencies:

```shell
git clone https://github.com/aiidateam/aiida-hubbard .
cd aiida-hubbard
pip install -e .[tests]  # install extra dependencies for test
pytest -sv tests # run tests
pytest -sv examples # run examples
```

You can also use `tox` to run the test set. Here you can also use the `-e` option to specify the Python version for the test run. Example:
```shell
pip install tox
tox -e py39 -- tests/calculations/hp/test_hp.py
```

### Pre-commit
To contribute to this repository, please enable pre-commit so the code in commits are conform to the standards.
Simply install the repository with the `pre-commit` extra dependencies:
```shell
cd aiida-hubbard
pip install -e .[pre-commit]
pre-commit install
```
