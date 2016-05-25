#!/usr/bin/env python
from cpyMSpec import IsotopePattern, centroidize
from pyimzml.ImzMLWriter import ImzMLWriter

from mz_axis import Instrument

import numpy as np

import cPickle
import argparse

np.random.seed(42)  # for reproducibility / debugging

parser = argparse.ArgumentParser(description="simulate a dataset from layers")
parser.add_argument('input', type=str, help="input file produced by assignMolecules.py")
parser.add_argument('output', type=str, help="output filename (centroided .imzML)")
parser.add_argument('--instrument', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
parser.add_argument('--res200', type=float, default=140000)

args = parser.parse_args()
instr = Instrument(args)

class SpectrumGenerator(object):
    def __init__(self, layers, mz_axis, detection_limit=1e-3):
        self.mz_axis = mz_axis
        self.layers = layers
        self.detection_limit = detection_limit

        # sparse noise parameters
        self.noise_prob = np.median(layers['noise']['prob'])
        self.noise_sqrt_avg = np.median(layers['noise']['sqrt_avg'])
        self.noise_sqrt_std = np.median(layers['noise']['sqrt_std'])

        # for each m/z bin, self._n_samples random intensity values are generated;
        self._n_samples = 50

        print "computing isotope patterns"
        self._computeIsotopePatterns()

        print "computing envelopes"
        self._computeEnvelopes()

    def _computeIsotopePatterns(self):
        self.isotope_patterns = {}

        for i in self.layers['layers_list'].keys():
            layer = self.layers['layers_list'][i]
            self.isotope_patterns[i] = []
            for sf in layer['sf_list']:
                data = {}
                data['p'] = p = IsotopePattern(sf['sf_a']).charged(1)
                data['resolution'] = instr.resolutionAt(p.masses[0])
                data['l'] = np.searchsorted(self.mz_axis, min(p.masses) - 0.5, 'l')
                data['r'] = np.searchsorted(self.mz_axis, max(p.masses) + 1, 'r')
                data['fwhm'] = p.masses[0] / data['resolution']
                data['intensity'] = sf['mult'][0]
                self.isotope_patterns[i].append(data)

    def _computeEnvelopes(self):
        self._envelopes = {}
        self._nnz = {}
        noisy_mzs = np.repeat(self.mz_axis, self._n_samples)

        # FIXME: make offset resolution-dependent / user-adjustable
        noisy_mzs += np.random.normal(scale=1e-4, size=len(noisy_mzs))

        for i in self.layers['layers_list'].keys():
            envelope = np.zeros_like(noisy_mzs)
            for d in self.isotope_patterns[i]:
                ln = max(0, (d['l'] - 1) * self._n_samples)
                rn = min(len(envelope), (d['r'] + 1) * self._n_samples)
                mzs = noisy_mzs[ln:rn]
                order = np.argsort(mzs)
                envelope_values = d['p'].envelope(d['resolution'])(mzs[order])
                envelope[ln:rn][order] += d['intensity'] * envelope_values
            # avoid storing zeros - they would occupy too much RAM
            self._nnz[i] = np.unique((np.where(envelope > 0)[0] / self._n_samples).astype(int))
            nnz_rnd = []
            for j in self._nnz[i]:
                nnz_rnd.extend([j * self._n_samples + k for k in xrange(self._n_samples)])
            self._envelopes[i] = envelope.take(nnz_rnd)

    def _addNoisyEnvelope(self, result, layer, x, y):
        layer_intensity = self.layers['layers_list'][layer]['image'][x, y]
        e = self._envelopes[layer]
        nnz = self._nnz[layer]
        idx = np.arange(len(nnz)) * self._n_samples
        idx += np.random.randint(0, self._n_samples, len(idx))
        result[nnz] += e[idx] * layer_intensity
        return result

    def _addSparseNoise(self, result, x, y):
        min_int = self.layers['min_intensities'][x, y]
        nmf_mz_axis = self.layers['nmf_mz_axis']
        bins = np.arange(len(nmf_mz_axis))
        noisy_bins = bins[np.random.binomial(2, self.noise_prob, len(bins)).astype(bool)]
        if not noisy_bins:
            return
        noise_mzs = nmf_mz_axis[noisy_bins, 0] + np.random.normal(0.0, 1e-3, len(noisy_bins))
        noise_intensities = np.random.normal(self.noise_sqrt_avg, self.noise_sqrt_std,
                                             len(noisy_bins))**2

        visible = noise_intensities > min_int
        bins = self.mz_axis[:-1].searchsorted(noise_mzs[visible])
        result[bins] += noise_intensities[visible]

    def generate(self, x, y, centroids=True):
        result = np.zeros_like(self.mz_axis)
        for i in self.layers['layers_list'].keys():
            self._addNoisyEnvelope(result, i, x, y)
        self._addSparseNoise(result, x, y)

        profile = (self.mz_axis, result)
        if centroids:
            p = centroidize(profile[0], profile[1])
            order = np.argsort(p.masses)
            masses = np.array(p.masses)[order]
            intensities = np.array(p.abundances)[order]

            # simulate limited dynamic range
            high_ints = intensities > self.detection_limit
            masses = masses[high_ints]
            intensities = intensities[high_ints]

            intensities *= max(profile[1])

            return masses, intensities
        else:
            return profile

with open(args.input) as f:
    layers = cPickle.load(f)

# FIXME: hardcoded mz_axis and detection_limit
sg = SpectrumGenerator(mz_axis=np.linspace(100, 1000, 1000000),
                       layers=layers,
                       detection_limit=1e-3)

def simulate_spectrum(sg, x, y):
    return sg.generate(x, y)

def writeSimulatedFile(spectrum_generator, output_filename):
    nx, ny = sg.layers['layers_list'][0]['image'].shape
    print nx, ny

    with ImzMLWriter(output_filename, mz_dtype=np.float32) as w:
        # steps can be increased to speed up the simulation
        # at expense of spatial resolution (for debugging purposes)
        step_x = 1
        step_y = 1

        for x in range(0, nx, step_x):
            for y in range(0, ny, step_y):
                if np.isinf(sg.layers['min_intensities'][x, y]):
                    continue
                mzs, intensities = simulate_spectrum(sg, x, y)
                w.addSpectrum(mzs, intensities, [x / step_x, y])
            print "{}% done".format(min(1.0, float(x + 1)/nx) * 100.0)

writeSimulatedFile(sg, args.output)
