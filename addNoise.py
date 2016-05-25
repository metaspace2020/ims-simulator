#!/usr/bin/env python

from cpyMSpec import centroidize
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="add noise to a clean dataset")
parser.add_argument('input', type=str, help="input file produced by simulateClean.py")
parser.add_argument('stats', type=str, help="statistics computed by collectStats.py")
parser.add_argument('output', type=str, help="output filename (centroided .imzML)")
parser.add_argument('--instrument', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
parser.add_argument('--res200', type=float, default=140000)

args = parser.parse_args()
# instr = Instrument(args)

class NoiseGenerator(object):
    # FIXME dummy generator, does only centroiding with detection limit
    def __init__(self, stats):
        self._sparsity_hist = stats['sparsityHist']
        self._intensity_hist = stats['intensityHist']

    def addNoise(self, profile_spectrum):
        spec = map(np.array, profile_spectrum)
        p = centroidize(*spec)
        mzs = np.array(p.masses)
        intensities = np.array(p.abundances) * spec[1].max()

        limit = 1e-3 * intensities.max()
        detectable = np.where(intensities > limit)[0]
        mzs = mzs[detectable]
        intensities = intensities[detectable]

        order = mzs.argsort()
        return mzs[order], intensities[order]

with open(args.stats) as stats:
    data = np.load(stats)
    ng = NoiseGenerator(data)

imzml = ImzMLParser(args.input)

with ImzMLWriter(args.output, mz_dtype=np.float32) as w:
    for i, coords in enumerate(imzml.coordinates):
        noisy_mzs, noisy_intensities = ng.addNoise(imzml.getspectrum(i))
        w.addSpectrum(noisy_mzs, noisy_intensities, coords)
