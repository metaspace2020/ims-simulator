#!/usr/bin/env python
from pyMSpec.pyisocalc import pyisocalc

import argparse
import cPickle

parser = argparse.ArgumentParser(description="assign molecules to peaks of NMF spectral components")
parser.add_argument('input', type=str, help="file with layers produced by assignMolecules.py")
parser.add_argument('output', type=str, help="text file containing all groundtruth molecules")

args = parser.parse_args()

def normalized(sf):
    return str(pyisocalc.parseSumFormula(sf))

with open(args.input) as f:
    layers = cPickle.load(f)

with open(args.output, "w+") as f:
    for layer in layers['layers_list']:
        for ion in layers['layers_list'][layer]['sf_list']:
            sf, adduct = ion['sf_a'].split('+')
            sf = normalized(sf)
            f.write("{},+{}\n".format(sf, adduct))
