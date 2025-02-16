---
myst:
    substitutions:
        aiida_pseudo: '[`aiida-pseudo`](https://aiida-pseudo.readthedocs.io/)'
        hubbard_structure: '{py:class}`~aiida_quantumespresso.data.hubbard_structure.HubbardStructureData`'
---

(howto-calculations-hp)=

# `hp.x`

The `hp.x` code of Quantum ESPRESSO performs a self-consistent perturbative calculation of Hubbard parameters
within Density-Functional-Perturbation Theory (DFPT),  using a plane-wave basis set and pseudopotentials (norm-conserving, ultra-soft and PAW).
This is a fundamental step to get accurate electronic properties of complex materials, mainly containing transition metals
for which the self-interaction error is relevant.

|                     |                                                               |
|---------------------|---------------------------------------------------------------|
| Plugin class        | {py:class}`~aiida_hubbard.calculations.hp.HpCalculation`  |
| Plugin entry point  | ``quantumespresso.hp``                                        |

:::{hint}
Remember that to exploit the best from the features of AiiDA to use the dedicated _WorkChains_,
which will provide this calculation with **automatic error handlings**.
Visit the [workflows](../workflows/index) section for more!
:::

## How to launch a `hp.x` calculation

Below is a script with a basic example of how to run a `hp.x` calculation through the `HpCalculation` plugin that computes the Hubbard
calculation of LiCoO{sub}`2`:

```{literalinclude} ../include/run_hp_basic.py
:language: python
```

Note that you may have to change the name of the codes (hp and pw) that is loaded using `load_code` and the pseudopotential family loaded with `load_group`.

:::{important}
The `hp.x` code needs to read the wavefunctions from a previously run `pw.x` calculation.
Thus, you need to first run a `PwCalculation` using the `HubbardStructureData` as input structure
with initialized Hubbard parameters to make `hp.x` understand which atoms to perturb.
You can find more information on how to do so on the [aiida-quantumespresso documentation](https://aiida-quantumespresso.readthedocs.io/en/latest/) .
Once this run is complete, you can move forward with the tutorial.
:::

:::{note}
In the provided script, the PwCalculation is performed before the HpCalculation.
:::

## How to define input file parameters

The `hp.x` code supports many parameters that can be defined through the input file,
as shown on the [official documentation](https://www.quantum-espresso.org/Doc/INPUT_HP.html).
The parameters are divided into a unique section or "card".
Parameters that are part of cards that start with an ampersand (`&`) should
be specified through the `parameters` input of the `HpCalculation` plugin.
The parameters are specified using a Python dictionary,
where each card is its own sub-dictionary, for example:

```python
parameters = {
    'INPUTHP': {
        'conv_thr_chi': 1.0e-6,
        'alpha_mix(10)': 0.1,
    },
}
```

The parameters dictionary should be wrapped in a {py:class}`~aiida.orm.nodes.data.dict.Dict` node
 and assigned to the `parameters` input of the process builder:

```python
from aiida.orm import Dict, load_code
builder = load_code('hp').get_builder()
parameters = {
    ...
}
builder.parameters = Dict(parameters)
```

:::{warning}
There are a number of input parameters that *cannot* be set, as they will be automatically set by the plugin based on other inputs, such as the `structure`.
These include:

- `INPUTHP.pseudo_dir`
- `INPUTHP.outdir`
- `INPUTHP.prefix`
- `INPUTHP.iverbosity`
- `INPUTHP.nq1`
- `INPUTHP.nq2`
- `INPUTHP.nq3`

Defining them anyway will result in an exception when launching the calculation.
:::

## How to define the ``pw.x`` (SCF) folder

Each `hp.x` calculation requires a previously run `PwCalculation` from which to take the wavefunctions and
other parameters. The relative folder can be specified in the `HpCalculation` plugin through the `parent_scf` input namespace.
This input takes a remote folder, instance of the {py:class}`~aiida.orm.RemoteFolder`.
For example, say you have successfully performed a `PwCalculation` (or equivantely `PwBaseWorkChain`) with PK 1, then:

```python
from aiida.orm import load_code

# The `remote_folder` stores the information of the
# relative path of the computer it was run on.
parent_scf = load_node(1).outputs.remote_folder

builder = load_code('hp').get_builder()
builder.parent_scf = parent_scf
```

## How to run a calculation with symlinking

Specify `PARENT_FOLDER_SYMLINK: False` in the `settings` input:

```python
builder = load_code('hp').get_builder()
builder.settings = Dict({'PARENT_FOLDER_SYMLINK': True})
```

If this setting is specified, the plugin will NOT symlink the SCF folder.
By default, this is set to `True` in order to save disk space.

## How to analyze the results

When a `HpCalculation` is completed, there are quite a few possible analyses to perform.

### How to inspect the final Hubbard parameters

A _complete_ `HpCalculation` will produce an {{ hubbard_structure }} containing the parsed Hubbard parameters.
The parameters are stored under the `hubbard` namespace:

```python
In [1]: node = load_node(HP_CALCULATION_IDENTIFIER)

In [2]: node.outputs.hubbard_structure.hubbard
Out[2]:
Hubbard(parameters=(HubbardParameters([...]), ...), projectors='ortho-atomic', formulation='dudarev')
```

To visualize them as Quantum ESPRESSO HUBBARD card:

```python
In [3]: from aiida_quantumespresso.utils.hubbard import HubbardUtils

In [4]: hubbard_card = HubbardUtils(node.outputs.hubbard_structure.hubbard).get_hubbard_card()

In [5]: print(hubbard_card)
Out[5]:
HUBBARD ortho-atomic
V Co-3d Co-3d 1 1 5.11
V Co-3d O-2p  1 2 1.65
...
```
