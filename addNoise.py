#!/usr/bin/env python

from cpyMSpec import centroidize
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import numpy as np
import cPickle

import argparse

parser = argparse.ArgumentParser(description="add noise to a clean dataset")
parser.add_argument('real', type=str, help="original centroided imzML")
parser.add_argument('simclean', type=str, help="input file produced by simulateClean.py")
parser.add_argument('nmf', type=str, help="factorization file produced by NNMF.py")
parser.add_argument('layers', type=str, help="layers file produces by assignMolecules.py")
parser.add_argument('output', type=str, help="output filename (centroided .imzML)")

args = parser.parse_args()

class NoiseGenerator(object):
    def __init__(self, nmf_fn, layers_fn, imzml_fn):
        self._imzml = ImzMLParser(imzml_fn)
        with np.load(nmf_fn) as data:
            nx, ny = data['shape']
            self._W = data['W'].reshape((nx, ny, -1))
            self._H = data['H']
            self._mz_axis = data['mz_axis']
        self._coords = {}
        for i, coords in enumerate(self._imzml.coordinates):
            self._coords[(coords[0], coords[1])] = i
        self._mz_bins = []
        for mz, ppm in self._mz_axis:
            self._mz_bins.append(mz * (1.0 + 1e-6 * ppm))

        # self._removeAssignedBins(layers_fn)

    def _removeAssignedBins(self, layers_fn):
        # buggy at the moment
        with open(layers_fn, 'rb') as f:
            layers = cPickle.load(f)
        for i in layers['layers_list']:
            assigned = layers['layers_list'][i]['assigned_mz_bins']
            assigned = assigned[assigned < self._H[i].shape[0]]
            print "#assigned bins in component #{}: {}".format(i + 1, len(assigned))
            h = np.zeros_like(self._H[i])
            h[assigned] = self._H[i][assigned]
            self._H[i] = h

    def _getRealSpectrum(self, x, y):
        return self._imzml.getspectrum(self._coords[(x, y)])

    def generateNoise(self, x, y):
        real_spectrum = self._getRealSpectrum(x, y)
        real_mzs, real_intensities = map(np.array, real_spectrum)

        min_mz, max_mz = self._mz_bins[0], self._mz_bins[-1]
        inside_range = (real_mzs >= min_mz) & (real_mzs <= max_mz)
        real_mzs = real_mzs[inside_range]
        real_intensities = real_intensities[inside_range]

        bins = np.digitize(real_mzs, self._mz_bins)
        n_bins = len(self._mz_bins)
        binned_real_intensities = np.bincount(bins, real_intensities, n_bins)
        binned_approx_intensities = self._W[x, y, :].dot(self._H)
        noise = np.abs(binned_real_intensities - binned_approx_intensities)
        # FIXME: avoid duplicating noise
        noise_intensities = noise[bins]
        noise_mzs = np.array(real_mzs)
        nnz = noise_intensities > min(real_intensities) / 2
        return noise_mzs[nnz], noise_intensities[nnz]

    def addNoise(self, profile_spectrum, coords):
        spec = map(np.array, profile_spectrum)
        p = centroidize(*spec)
        mzs = np.array(p.masses)
        intensities = np.array(p.abundances) * spec[1].max()

        noise_mzs, noise_intensities = self.generateNoise(*coords)
        mzs = np.concatenate([mzs, noise_mzs])
        intensities = np.concatenate([intensities, noise_intensities])

        limit = min(self._getRealSpectrum(*coords)[1])
        detectable = np.where(intensities > limit)[0]
        mzs = mzs[detectable]
        intensities = intensities[detectable]

        order = mzs.argsort()
        return mzs[order], intensities[order]

ng = NoiseGenerator(args.nmf, args.layers, args.real)

imzml_sim = ImzMLParser(args.simclean)

with ImzMLWriter(args.output, mz_dtype=np.float32) as w:
    for i, coords in enumerate(imzml_sim.coordinates):
        noisy_mzs, noisy_intensities = ng.addNoise(imzml_sim.getspectrum(i), coords)
        w.addSpectrum(noisy_mzs, noisy_intensities, coords)
