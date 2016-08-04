Scripts for simulating high-resolution imaging mass spectrometry datasets.

# Usage

## Installation of used tools/packages

1. Install miniconda for Python 2.7 (http://conda.pydata.org/miniconda.html)
2. Run `conda env create` which will grab all necessary packages
3. Run `source activate ims_simulator`

## How to get a simulated .imzML from a real .imzML

1. Create a working directory where all generated files will be kept (`mkdir <dirname> && cd <dirname>`).
2. Copy `example_config.yaml` to the newly created directory and edit it as needed. The most important bit is to specify the .imzML filename.
3. Run `python <path to ims-simulator>/pipeline.py <your config.yaml>` from the directory and wait. In about an hour it should successfully finish.
4. If the run completes without errors, you will find a file named `report_<config hash>.pdf` along with the generated `imzML` file. It contains some useful metrics for comparing simulated and original datasets.

# License

Unless specified otherwise in file headers, all files are licensed under Apache 2.0 license.
