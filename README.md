# Lorentz Invariance Violation with Neutrinos and Gamma-ray bursts

### Installation

The package can be installed with `pip install -e .` (preferrably in a virtual environment) and used directly.

### Structure

There are four main modules:
* `inputs`: handles the formatting of neutrino data and GRB catalog, including the computation of the PDFs, scrambling, signal injection...
* `model`: defines the functions related to the LIV, especially the time delay and the time PDF (computed from the PDFs).
* `statistics`: defines the likelihood and all related quantities, computes the TS, best fit, 1D intervals, and 2D contours.
* `run`: defines the functions to run the actual analyses (see usage below).

### Usage

##### Existing macros

A function to generate pseudo-experiments is ready to use, as defined in `run.py` and available directly as script after package installation:

* `liv-run-ts [-h] [--config CONFIG] [--npes NPES] [--fsig FSIG] [--nsig NSIG] [--eta ETA]`
  Compute the TS for scrambled data or data with some signal injection.
```bash
optional arguments:
  --config         Path to the configuration file.
  --npes NPES      Number of PEs to be generated. Default: 100.
  --fsig FSIG      Injected fraction of signal events. Default: 0.
  --nsig NSIG      Number of injected signal events. Default: 0.
  --eta ETA        Injected eta. Default: 0.
```

##### How to start from scratch

You can also play with the different functions directly using the following functions:

1. Build the GRB catalog:
```python
from living.inputs.grb import GRBCatalog
grbs = GRBCatalog()
grbs.load_csv("inputs/grb_catalog.csv")
```

2. Build the dataset:
```python
from living.inputs.icecube import ICHESEDataset
dataset = ICHESEDataset()
dataset.compute_pdfs()
```

3. Apply scrambling or do some signal injection:
```python
grbs_pe = grbs.scramble()
data_pe = dataset.scramble()
# OR
data_pe = dataset.inject_signal(fsig=0.01, eta=100, grbs_pe)
```

4. Get list of data events from the above objects, doing some precomputations:
```python
events = data_pe.events(dataset, grbs_pe)
```

5. Do anything you may want:
```python
from living.statistics import test_statistic, bestfit
print("TS: ", test_statistic(events, dataset, grbs_pe)[0])
bf_fsig, bf_eta = bestfit(events, dataset, grbs_pe)
print("Best-fit point:", bestfit(events, dataset, grbs_pe))
```

##### Example scripts

Scripts to analyse the results and produce some example plots are available in `scripts/`. Especially, they may be used to perform the analysis of IceCube-HESE 12yr dataset.