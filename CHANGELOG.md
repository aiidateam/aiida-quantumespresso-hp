## v0.1.0:
First official release of `aiida-hubbard`, the official plugin for the HP code of Quantum ESPRESSO to the AiiDA platform.
The following calculations, parsers, and workflows are provided:

### Calculations
- `HpCalculation`: calculation plugin for `hp.x`

### Parsers
- `HpParser`: parser for the `hp.x` calculation

### Workflows
- `HpBaseWorkChain`: workflow to run a `HpCalculation` to completion
- `HpParallelizeAtomsWorkChain`: workflow to parallelize an `hp.x` calculation as independent atoms child subprocesses
- `HpParallelizeQpointsWorkChain`: workflow to parallelize an `HpParallelizeAtomsWorkChain` calculation as independent q-points child subprocesses
- `HpWorkChain`: workflow to run a manage parallel capabilities of `hp.x`, by proprerly calling the correct workchains
- `SelfConsistentHubbardWorkChain`: worfklow to calculate self-consistently the Hubbard parameters with on-the-fly nearest neighbours detection
