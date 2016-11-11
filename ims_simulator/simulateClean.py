#!/usr/bin/env python
from cpyMSpec import IsotopePattern
from pyimzml.ImzMLWriter import ImzMLWriter

from mz_axis import Instrument
from adduct import splitSumFormula, adductCharge

import numpy as np

import cPickle
import argparse

parser = argparse.ArgumentParser(description="simulate a clean (profile) dataset from layers")
parser.add_argument('input', type=str, help="input file produced by assignMolecules.py")
parser.add_argument('output', type=str, help="output filename (centroided .imzML)")
parser.add_argument('--instrument', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
parser.add_argument('--res200', type=float, default=140000)

args = parser.parse_args()
instr = Instrument(args)

class SpectrumGenerator(object):
    def __init__(self, layers, mz_axis):
        self.mz_axis = mz_axis
        self.layers = layers

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
                _, a = splitSumFormula(sf['sf_a'])
                charge = adductCharge(a)
                data['p'] = p = IsotopePattern(sf['sf_a']).charged(charge)
                data['resolution'] = instr.resolutionAt(p.masses[0])
                data['l'] = np.searchsorted(self.mz_axis, min(p.masses) - 0.5, 'l')
                data['r'] = np.searchsorted(self.mz_axis, max(p.masses) + 1, 'r')
                data['fwhm'] = p.masses[0] / data['resolution']
                data['intensity'] = sf['mult']
                self.isotope_patterns[i].append(data)

    def _computeEnvelopes(self):
        self._envelopes = {}
        self._nnz = {}

        for i in self.layers['layers_list'].keys():
            envelope = np.zeros_like(self.mz_axis)
            for d in self.isotope_patterns[i]:
                l, r = d['l'], d['r']
                mzs = self.mz_axis[l:r]
                envelope_values = d['p'].envelope(d['resolution'])(mzs)
                envelope[l:r] += d['intensity'] * envelope_values
            # avoid storing zeros - they would occupy too much RAM
            self._nnz[i] = np.where(envelope > 0)[0]
            self._envelopes[i] = envelope.take(self._nnz[i])

    def _addEnvelope(self, result, layer, x, y):
        layer_intensity = self.layers['layers_list'][layer]['image'][x, y]
        e = self._envelopes[layer]
        nnz = self._nnz[layer]
        idx = np.arange(len(nnz))
        result[nnz] += e[idx] * layer_intensity
        return result

    def generate(self, x, y, centroids=True):
        result = np.zeros_like(self.mz_axis)
        for i in self.layers['layers_list'].keys():
            self._addEnvelope(result, i, x, y)

        # don't store zeross
        nnz = result > (1e-6 * result.max()) #Q: should be related to dynamic range?
        return (self.mz_axis[nnz], result[nnz])

with open(args.input) as f:
    layers = cPickle.load(f)

# FIXME: hardcoded mz_axis
sg = SpectrumGenerator(mz_axis=np.linspace(100, 1000, 1000000),
                       layers=layers)

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
                w.addSpectrum(mzs, intensities, [x / step_x + 1, y + 1])
            print "{}% done".format(min(1.0, float(x + 1)/nx) * 100.0)

writeSimulatedFile(sg, args.output)
