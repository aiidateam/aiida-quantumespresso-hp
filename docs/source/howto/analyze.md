(howto-analyze)=

# How to analyze the results

When a `SelfConsistentHubbardWorkChain` is completed, there are quite a few possible analyses to perform.

## How to inspect the final Hubbard parameters

A _complete_ `SelfConsistentHubbardWorkChain` will produce a {{ hubbard_structure }} containing the parsed Hubbard parameters.
The parameters are stored under the `hubbard` namespace:

```shell
In [1]: node = load_node(HP_CALCULATION_IDENTIFIER)

In [2]: node.outputs.hubbard_structure.hubbard
Out[2]:
Hubbard(parameters=(HubbardParameters([...]), ...), projectors='ortho-atomic', formulation='dudarev')
```

This corresponds to a `pydantic` class, so you can access the stores values (`parameters`, `projectors`, `formulations`) simply by:
```shell
In [3]: node.outputs.hubbard_structure.hubbard.parameters
Out[3]: [HubbardParameters(atom_index=0, atom_manifold='3d', neighbour_index=0, neighbour_manifold='3d', translation=(0, 0, 0), value=5.11, hubbard_type='Ueff'), ...]
```

To access to a specific value:
```shell
In [4]: hubbard_structure.hubbard.parameters[0].value
Out[4]: 5.11
```

To visualize them as Quantum ESPRESSO HUBBARD card:

```shell
In [5]: from aiida_quantumespresso.utils.hubbard import HubbardUtils

In [6]: hubbard_card = HubbardUtils(node.outputs.hubbard_structure.hubbard).get_hubbard_card()

In [7]: print(hubbard_card)
Out[7]:
HUBBARD ortho-atomic
V Co-3d Co-3d 1 1 5.11
V Co-3d O-2p  1 2 1.65
...
```
