# `aiida-hubbard`

AiiDA plugin for the first-principles calculation of Hubbard parameters.

This is also the official AiiDA plugin for the [HP](https://www.sciencedirect.com/science/article/pii/S0010465522001746) code of [Quantum ESPRESSO](https://www.quantum-espresso.org/).

|    | |
|-----|----------------------------------------------------------------------------|
|Latest release| [![PyPI version](https://badge.fury.io/py/aiida-hubbard.svg)](https://badge.fury.io/py/aiida-hubbard)[![PyPI pyversions](https://img.shields.io/pypi/pyversions/aiida-hubbard.svg)](https://pypi.python.org/pypi/aiida-hubbard) |
|Getting help| [![Docs status](https://readthedocs.org/projects/aiida-hubbard/badge)](http://aiida-hubbard.readthedocs.io/) [![Discourse status](https://img.shields.io/discourse/status?server=https%3A%2F%2Faiida.discourse.group%2F)](https://aiida.discourse.group/)
|Build status| [![Build Status](https://github.com/aiidateam/aiida-hubbard/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/aiidateam/aiida-hubbard/actions) [![Coverage Status](https://codecov.io/gh/aiidateam/aiida-hubbard/branch/master/graph/badge.svg)](https://codecov.io/gh/aiidateam/aiida-hubbard) |
|Activity| [![PyPI-downloads](https://img.shields.io/pypi/dm/aiida-hubbard.svg?style=flat)](https://pypistats.org/packages/aiida-hubbard) [![Commit Activity](https://img.shields.io/github/commit-activity/m/aiidateam/aiida-hubbard.svg)](https://github.com/aiidateam/aiida-hubbard/pulse)
|Community|  [![Discourse](https://img.shields.io/discourse/topics?server=https%3A%2F%2Faiida.discourse.group%2F&logo=discourse)](https://aiida.discourse.group/)

## Compatibility matrix

The matrix below assumes the user always install the latest patch release of the specified minor version, which is recommended.

| Plugin | AiiDA | Python | Quantum ESPRESSO |
|-|-|-|-|
| `v0.1.0` | ![Compatibility for v4.0][AiiDA v4.0-pydantic2] |  [![PyPI pyversions](https://img.shields.io/pypi/pyversions/aiida-hubbard.svg)](https://pypi.python.org/pypi/aiida-hubbard) | ![Quantum ESPRESSO compatibility][QE v7.2-7.4] |

## Installation
To install using pip, simply execute:

    pip install aiida-hubbard

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


[AiiDA v4.0-pydantic2]: https://img.shields.io/badge/AiiDA->=2.3.0,<2.6.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAhCAYAAABTERJSAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAFhgAABYYBG6Yz4AAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAUbSURBVFiFzZhrbFRVEMd%2Fc%2B5uu6UUbIFC%2FUAUVEQCLbQJBIiBDyiImJiIhmohYNCkqJAQxASLF8tDgYRHBLXRhIcKNtFEhVDgAxBJqgmVh4JEKg3EIn2QYqBlt917xg%2BFss%2ByaDHOtzsz5z%2B%2FuZl7ztmF%2F5HJvxVQN6cPYX8%2FPLnOmsvNAvqfwuib%2FbNIk9cQeQnLcKRL5xLIV%2Fic9eJeunjPYbRs4FjQSpTB3aS1IpRKeeOOewajy%2FKKEO8Q0DuVdKy8IqsbPulxGHUfCBBu%2BwUYGuFuBTK7wQnht6PEbf4tlRomVRjCbXNjQEB0AyrFQOL5ENIJm7dTLZE6DPJCnEtFZVXDLny%2B4Sjv0PmmYu1ZdUek9RiMgoDmJ8V0L7XJqsZ3UW8YsBOwEeHeeFce7jEYXBy0m9m4BbXqSj2%2Bxnkg26MCVrN6DEZcwggtd8pTFx%2Fh3B9B50YLaFOPwXQKUt0tBLegtSomfBlfY13PwijbEnhztGzgJsK5h9W9qeWwBqjvyhB2iBs1Qz0AU974DciRGO8CVN8AJhAeMAdA3KbrKEtvxhsI%2B9emWiJlGBEU680Cfk%2BSsVqXZvcFYGXjF8ABVJ%2BTNfVXehyms1zzn1gmIOxLEB6E31%2FWBe5rnCarmo7elf7dJEeaLh80GasliI5F6Q9cAz1GY1OJVNDxTzQTw7iY%2FHEZRQY7xqJ9RU2LFe%2FYqakdP911ha0XhjjiTVAkDwgatWfCGeYocx8M3glG8g8EXhSrLrHnEFJ5Ymow%2FkhIYv6ttYUW1iFmEqqxdVoUs9FmsDYSqmtmJh3Cl1%2BVtl2s7owDUdocR5bceiyoSivGTT5vzpbzL1uoBpmcAAQgW7ArnKD9ng9rc%2BNgrobSNwpSkkhcRN%2BvmXLjIsDovYHHEfmsYFygPAnIDEQrQPzJYCOaLHLUfIt7Oq0LJn9fxkSgNCb1qEIQ5UKgT%2Fs6gJmVOOroJhQBXVqw118QtWLdyUxEP45sUpSzqP7RDdFYMyB9UReMiF1MzPwoUqHt8hjGFFeP5wZAbZ%2F0%2BcAtAAcji6LeSq%2FMYiAvSsdw3GtrfVSVFUBbIhwRWYR7yOcr%2FBi%2FB1MSJZ16JlgH1AGM3EO2QnmMyrSbTSiACgFBv4yCUapZkt9qwWVL7aeOyHvArJjm8%2Fz9BhdI4XcZgz2%2FvRALosjsk1ODOyMcJn9%2FYI6IrkS5vxMGdUwou2YKfyVqJpn5t9aNs3gbQMbdbkxnGdsr4bTHm2AxWo9yNZK4PXR3uzhAh%2BM0AZejnCrGdy0UvJxl0oMKgWSLR%2B1LH2aE9ViejiFs%2BXn6bTjng3MlIhJ1I1TkuLdg6OcAbD7Xx%2Bc3y9TrWAiSHqVkbZ2v9ilCo6s4AjwZCzFyD9mOL305nV9aonvsQeT2L0gVk4OwOJqXXVRW7naaxswDKVdlYLyMXAnntteYmws2xcVVZzq%2BtHPAooQggmJkc6TLSusOiL4RKgwzzYU1iFQgiUBA1H7E8yPau%2BZl9P7AblVNebtHqTgxLfRqrNvZWjsHZFuqMqKcDWdlFjF7UGvX8Jn24DyEAykJwNcdg0OvJ4p5pQ9tV6SMlP4A0PNh8aYze1ArROyUNTNouy8tNF3Rt0CSXb6bRFl4%2FIfQzNMjaE9WwpYOWQnOdEF%2BTdJNO0iFh7%2BI0kfORzQZb6P2kymS9oTxzBiM9rUqLWr1WE5G6ODhycQd%2FUnNVeMbcH68hYkGycNoUNWc8fxaxfwhDbHpfwM5oeTY7rUX8QAAAABJRU5ErkJggg%3D%3D

[QE v7.2-7.4]: https://img.shields.io/badge/Quantum%20ESPRESSO-%3E=7.2,%3C=7.4-007ec6.svg?logo=data:image/png;base64,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
