Scripts for simulating high-resolution imaging mass spectrometry datasets.

# Usage

## Installation of used tools/packages

1. Install miniconda for Python 2.7 (http://conda.pydata.org/miniconda.html)
2. Run `conda env create` which will grab all necessary packages
3. Run `source activate ims_simulator`

## Pipeline steps to get a simulated .imzML from a real .imzML

1. Run `ims convert <real.imzML> <real.imzb>` to get an m/z-sorted file in an internal format
2. Run NNMF.py script to produce a numpy-readable NMF factorization from `<real.imzb>`
3. Run assignMolecules.py script to assign some molecules to the peaks from NMF factorization
4. Run simulate.py to generate an output .imzML from the layers produced in the previous step

# License

Unless specified otherwise in file headers, all files are licensed under Apache 2.0 license.
