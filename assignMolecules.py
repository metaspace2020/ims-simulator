#!/usr/bin/env python
import os
import subprocess

import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

from pyMSpec.pyisocalc import pyisocalc
from cpyMSpec import IsotopePattern, centroidize

from mz_axis import Instrument

import argparse

parser = argparse.ArgumentParser(description="assign molecules to peaks of NMF spectral components")
parser.add_argument('input', type=str, help="input file produced by NNMF.py")
parser.add_argument('output', type=str, help="output file with images and molecules (.pkl)")
parser.add_argument('--instrument', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
parser.add_argument('--res200', type=float, default=140000)
parser.add_argument('--db', type=str, help="text file with desired molecules, one per line")
parser.add_argument("--dynrange", type=float, default=1000, help="dynamic range: influences how many peaks in each component will be annotated")

args = parser.parse_args()
instr = Instrument(args)
assert(args.dynrange > 1)
detection_limit = 1.0 / args.dynrange

output_filename = os.path.join(os.getcwd(), os.path.expanduser(args.output))

db = set()
if args.db:
    for line in open(args.db):
        sf_str = str(pyisocalc.parseSumFormula(line.strip()))
        db.add(sf_str)
    print "target database size:", len(db)

layers = {}
noise = {}

with open(args.input) as f:
    with np.load(f) as data:
        W = data['W']
        H = data['H']
        shape = data['shape']
        mz_axis = data['mz_axis'][:, 0]
        nmf_ppms = data['mz_axis'][:, 1]
        noise['prob'] = data['noise_prob']
        noise['sqrt_avg'] = data['noise_sqrt_avg']
        noise['sqrt_std'] = data['noise_sqrt_std']
        layers['noise'] = noise
        layers['min_intensities'] = data['min_intensities'][1:, 1:]
        layers['nmf_mz_axis'] = data['mz_axis']

def _combine_results(dfs):
    if len(dfs) == 0:
        return []

    df = pd.concat(dfs)
    df['abs_ppm'] = abs(df['ppm'])
    df = df.sort_values(by='abs_ppm')

    df = df[df['unsat'] >= 0]

    del df['abs_ppm']
    df.index = range(len(df))

    return df[['em', 'error', 'mf', 'ppm', 'unsat', 'adduct']]

# FIXME: make it work for negative mode
def search_mz_candidates_pfg(mass, adducts, ppm_limit=5, charge=1):
    dfs = []
    for adduct in adducts:
        mass_ = mass - IsotopePattern(adduct).charged(charge).masses[0]
        cmd_line = ("OMP_NUM_THREADS=2 PFG -m {} -t {} " +
                    "--C 0-100 --H 0-100 --N 0-10 --O 0-10 --S 0-5 --P 0-5 -r 'lewis'").format(mass_, ppm_limit)
        _ = subprocess.check_output(cmd_line, shell=True)
        results = open("result.txt").readlines()[1:]
        results = zip(*[s.split() for s in results])
        if not results:
            continue
        formula, _, mz, error, dbe = results

        df = pd.DataFrame.from_dict({'mf': np.array(formula, dtype=str),
                                     'em': np.array(mz, dtype=float),
                                     'unsat': np.array(dbe, dtype=float).astype(int)
                                     })

        df['error'] = df['em'] - mass_
        df['ppm'] = df['error'] / mass_ * 1e6

        df['adduct'] = adduct
        dfs.append(df)

    return _combine_results(dfs)

class AnnotatedSpectrum(object):
    def __init__(self, mzs_c, ints_c, ppms):
        order = np.argsort(np.abs(ints_c))
        self.mzs = np.asarray(mzs_c)[order]
        self.ppms = np.asarray(ppms)[order]
        self.ints = np.asarray(ints_c)[order]
        self.orig_mzs = self.mzs
        self._assigned_mzs = []

    def trim(self, min_intensity):
        self.mzs = self.mzs[self.ints >= detection_limit]
        self.ppms = self.ppms[self.ints >= detection_limit]
        self.ints = self.ints[self.ints >= detection_limit]

    def top(self):
        if len(self.mzs) == 0:
            return (0, 0)
        return self.mzs[-1], self.ppms[-1], self.ints[-1]

    def pop(self):
        self.mzs = self.mzs[:-1]
        self.ints = self.ints[:-1]
        self.ppms = self.ppms[:-1]

    def assign(self, isotope_pattern, ppm):
        theor_spec = isotope_pattern
        prev_intensity = None
        keep_list_all = np.ones_like(self.mzs, dtype=np.bool)
        for m, a in zip(theor_spec.masses, theor_spec.abundances):
            keep_list = 1e6 * np.abs(self.mzs - m) / m > ppm
            keep_list_all &= keep_list
            peak_list = np.where(np.logical_not(keep_list))[0]
            if len(peak_list) == 0:
                break
            intensity = self.ints[peak_list].sum()
            if prev_intensity and abs(intensity) > abs(prev_intensity):
                break
            prev_intensity = intensity

        if np.all(keep_list_all):
            print 'wtf'
            self.pop()
            return False
        else:
            self._assigned_mzs.append(self.mzs[np.logical_not(keep_list_all)])
            self.mzs = self.mzs[keep_list_all]
            self.ppms = self.ppms[keep_list_all]
            self.ints = self.ints[keep_list_all]
            return True

    def assignedMzBins(self):
        assigned_mzs = np.concatenate(self._assigned_mzs)
        return np.searchsorted(self.orig_mzs, assigned_mzs)

class MoleculeAssigner(object):
    def __init__(self, target_database):

        # keep track of the already assigned sum formulas
        # so that if the same m/z appears in different components,
        # it (hopefully) gets assigned the same sum formula
        self._already_assigned = set()

        # scoring the candidates for a given m/z incorporates our desire
        # to have some of the selected sum formulas from a predefined database;
        self._target_db = target_database

    def normalize_sf(self, sf):
        return str(pyisocalc.parseSumFormula(sf))

    def _score_match(self, row, ppm_t):
        normalized_mf = self.normalize_sf(row.mf)
        if normalized_mf in self._target_db:
            return 5
        if normalized_mf in self._already_assigned:
            return 10
        return 1

    def _centroids(self, row):
        p = IsotopePattern("{}+{}".format(row.mf, row.adduct))
        p = p.centroids(instr.resolutionAt(p.masses[0]))
        p = p.charged(1).trimmed(5)
        return p

    def fit_spectrum(self, mzs_c, ppms, ints_c, detection_limit, max_peaks=500):
        annotated_spec = AnnotatedSpectrum(mzs_c, ints_c, ppms)
        annotated_spec.trim(detection_limit)
        annotated_spec.sf_list = []

        while annotated_spec.top()[1] >= detection_limit:
            mz, ppm, v = annotated_spec.top()
            hits = search_mz_candidates_pfg(mz, ['H', 'Na', 'K'], ppm_limit=ppm)
            if len(hits) == 0:
                annotated_spec.pop()
                continue

            scores = [self._score_match(candidate, ppm)
                      for candidate in hits.itertuples()]

            best_hit_idx = np.argmax(scores)
            best_hit = hits.iloc[best_hit_idx]

            theor_spec = self._centroids(best_hit)
            if not annotated_spec.assign(theor_spec, ppm):
                continue

            annotated_spec.sf_list.append([v, best_hit])
            self._already_assigned.add(self.normalize_sf(best_hit['mf']))

            if len(annotated_spec.sf_list) > max_peaks:
                break

        return annotated_spec

assigner = MoleculeAssigner(db)

spec_fit = []
db_percentage = []
for ii in range(H.shape[0]):
    print "coeff", ii
    coeff_spec = H[ii]
    fit = assigner.fit_spectrum(mz_axis, nmf_ppms, coeff_spec, detection_limit)
    spec_fit.append(fit)

    sum_formulas = set([assigner.normalize_sf(x[1]['mf']) for x in fit.sf_list])

    if len(fit.sf_list) > 0:
        db_percentage.append(len(db & sum_formulas) / float(len(fit.sf_list)))
        print "database percentage in component", ii+1, ":", db_percentage[-1]

print "median percentage of database formulas:", np.median(db_percentage)

layers['layers_list'] = OrderedDict()
for ii in range(len(spec_fit)):
    layers['layers_list'][ii] = {}
    layers['layers_list'][ii]['assigned_mz_bins'] = spec_fit[ii].assignedMzBins()
    layers['layers_list'][ii]['image'] = W[:, ii].reshape(shape)[1:, 1:]
    layers['layers_list'][ii]['sf_list'] = []
    for sf in spec_fit[ii].sf_list:
        sf_a = "{}+{}".format(sf[1]['mf'], sf[1]['adduct'])
        mult = sf[0]  # intensity
        layers['layers_list'][ii]['sf_list'].append({"sf_a": sf_a, "mult": mult})

# save the layers for later analysis
with open(output_filename, "w+") as f:
    pickle.dump(layers, f)
