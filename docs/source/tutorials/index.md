(tutorials)=

# Tutorials

:::{important}
Before you get started, make sure that you have:

- installed the `aiida-quantumespresso` package ([see instructions](installation-installation))
- configured the `pw.x` *AND* the `hp.x` codes ([see instructions](installation-setup-code))
- installed the SSSP pseudopotential family ([see instructions](installation-setup-pseudopotentials))
:::

In this section you will find some tutorials that you will guide you through how to use the aiida-hubbard plugin, from **zero** to **hero**! We strongly recommend to start from the first one and work your way up with the other ones.

Go to one of the tutorials!

1. [Computing Hubbard parameters](../1_computing_hubbard.ipynb): get started with predicting the Hubbard parameters step by step, by using the _WorkChains_ of the aiida-quantumespresso(-hp) suite.
2. [Hubbard parameters in parallel](../2_parallel_hubbard.ipynb): learn the automated parallel calculation of Hubbard parameters to speed up your work.
3. [Self-consistent Hubbard parameters](../3_self_consistent.ipynb): compute self-consistently the Hubbard parameters in automated fashion.

Here below the estimated time to run each tutorial (jupyter notebook):

```{nb-exec-table}
```

```{toctree}
:maxdepth: 1
:hidden: true

../1_computing_hubbard.ipynb
../2_parallel_hubbard.ipynb
../3_self_consistent.ipynb
```
