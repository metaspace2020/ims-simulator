Scripts for simulating high-resolution imaging mass spectrometry datasets.

# Usage
## Prerequisites

1. miniconda (see [Install miniconda for Python 2.7](http://conda.pydata.org/miniconda.html))
2. git (see [Getting Started Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

## Installation

1. Create a directory for this repository 
 
 ```
 mkdir ~/projects/ims_simulator/
 ```
2. Clone this repository
   
   ```
   cd ~/projects/ims_simulator/
   git clone https://github.com/SpatialMetabolomics/ims-simulator.git
   ```
3. Set up a virtual environment for the code
  
  ```
  cd ims-simulator/
  conda env create
  source activate ims_simulator
  ```

## How to get a simulated .imzML from a real .imzML

1. Create a working directory where all generated files will be kept (`mkdir -p ~/projects/ims_simulator/data/test/ && cd ~/projects/ims_simulator/data/test/`).
2. Copy `example_config.yaml` from to the newly created directory `cp ~/projects/ims_simulator/ims-simulator/example_config.yaml ~/projects/ims_simulator/data/test/config.yaml`
3. Open the `config.yaml` file in your favourite text editor and change the `imzml:` field to be a centroided .imzml dataset you wish to use as a template for the simulation (the default is an example file provided along with this repository) 
3. Run `python ~/projects/ims_simulator/ims-simulator/pipeline.py ~/projects/ims_simulator/data/test/config.yaml` from the directory and wait. In about an hour it should successfully finish.
4. If the run completes without errors, you will find a file named `report_<config hash>.pdf` along with the generated `imzML` file. It contains some useful metrics for comparing simulated and original datasets.

# License

Unless specified otherwise in file headers, all files are licensed under Apache 2.0 license.
