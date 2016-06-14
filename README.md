Scripts for simulating high-resolution imaging mass spectrometry datasets.

# Usage

## Installation of used tools/packages

1. Install miniconda for Python 2.7 (http://conda.pydata.org/miniconda.html)
2. Run `conda env create` which will grab all necessary packages
3. Run `source activate ims_simulator`
4. Install PFG from https://github.com/zmzhang/PFG (**FIXME:** package it for Anaconda Cloud)
   1.  Clone PFG e.g. 
   
        ```bash
        mkdir pfg
        cd pfg
        git clone https://github.com/zmzhang/PFG
        ```
   2. install
      ```bash
      cd PFG
      make
      ```
      Note: on Mac install gcc-5 with ``brew install gcc``` and edit Makefile 
      
      ```bash
      CC          = gcc-5
      CXX         = g++-5
      ```
5. Edit assignMolecule.py and set ```pfg_executable_dir = os.path.expanduser("../pfg/PFG")``` to the PFG directory

## How to get a simulated .imzML from a real .imzML

1. Copy `example_config.yaml` and edit it as needed. The most important bit is to specify the .imzML filename.
2. Run `python pipeline.py <your config.yaml>` and wait. In about an hour it should successfully finish.

# License

Unless specified otherwise in file headers, all files are licensed under Apache 2.0 license.

