Scripts for simulating high-resolution imaging mass spectrometry datasets.

# Usage

## Installation of used tools/packages

1. Install miniconda for Python 2.7 (http://conda.pydata.org/miniconda.html)
2. Run `conda env create` which will grab all necessary packages
3. Run `source activate ims_simulator`
4. Install [PFG](https://github.com/zmzhang/PFG) manually (**FIXME:** package it for Anaconda Cloud)

## How to get a simulated .imzML from a real .imzML

1. Copy `example_config.yaml` and edit it as needed. The most important bit is to specify the .imzML filename.
2. Run `python pipeline.py <your config.yaml>` and wait. In about an hour it should successfully finish.

# License

Unless specified otherwise in file headers, all files are licensed under Apache 2.0 license.
