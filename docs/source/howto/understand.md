(howto-understand)=

# How-to understand the input/builder structure

In AiiDA the CalcJobs and WorkChains have usually nested inputs and different options on how to run the calculation
and/or workflows. To understand the nested input structure of CalcJobs/Workflows, we made them available in a clickable
fashion in the [topics section](topics).

Moreover, it could be useful to understand the
[_expose inputs/outputs_](https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/workflows/usage.html#modular-workflow-design)
mechanism used in AiiDA for workflows, which guarantees a __modular design__.
This means that the workflows can use the inputs of other workflows or calculations, and specify them under a new namespace.

This is the case for many workflows in this package. For example, the {class}`~aiida_hubbard.workflows.hubbard.SelfConsistentHubbardWorkChain` makes use of three WorkChains, the `PwBaseWorkChain` for the scf calculation (namespace used is `scf`),
the `PwRelaxWorkChain` for the (vc)relaxation part (namespace used is `relax`), and finally the `HpWorkChain` (namespace used is `hubbard`).
