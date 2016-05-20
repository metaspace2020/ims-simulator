#!/usr/bin/env python
import os
import subprocess

import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

from pyMSpec.pyisocalc import pyisocalc
from cpyMSpec import IsotopePattern

import argparse

parser = argparse.ArgumentParser(description="assign molecules to peaks of NMF spectral components")
parser.add_argument('input', type=str, help="input file produced by NNMF.py")
parser.add_argument('output', type=str, help="output file with images and molecules (.pkl)")
parser.add_argument('--instrument', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
parser.add_argument('--res200', type=float, default=140000)
parser.add_argument('--db', type=str, help="text file with desired molecules, one per line")

args = parser.parse_args()

output_filename = os.path.join(os.getcwd(), os.path.expanduser(args.output))

db = set()
if args.db:
    for line in open(args.db):
        sf_str = str(pyisocalc.parseSumFormula(line.strip()))
        db.add(sf_str)
    print "target database size:", len(db)

# FIXME: put it to conda or use chemcalc
pfg_executable_dir = os.path.expanduser("~/github/PFG")

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
    os.chdir(pfg_executable_dir)

    dfs = []
    for adduct in adducts:
        mass_ = mass - IsotopePattern(adduct).charged(charge).masses[0]
        cmd_line = ("OMP_NUM_THREADS=2 ./PFG -m {} -t {} " +
                    "--C 0-100 --H 0-100 --N 0-10 --O 0-10 --S 0-5 --P 0-5 -r 'lewis'").format(mass_, ppm_limit)
        _ = subprocess.check_output(cmd_line, shell=True)
        results = open(os.path.join(pfg_executable_dir, "result.txt")).readlines()[1:]
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

# FIXME code duplication with NNMF
def resolutionAt(mz):
    if args.instrument == 'orbitrap':
        return args.res200 * (200.0 / mz) ** 0.5
    elif args.instrument == 'fticr':
        return args.res200 * (200.0 / mz)

class MoleculeAssigner(object):
    def __init__(self, target_database):

        # keep track of the already assigned sum formulas
        # so that if the same m/z appears in different components,
        # it (hopefully) gets assigned the same sum formula
        self._already_assigned = set()

        # scoring the candidates for a given m/z incorporates our desire
        # to have some of the selected sum formulas from a predefined database;
        self._target_db = target_database

    def _normalize_sf(self, sf):
        return str(pyisocalc.parseSumFormula(sf))

    def _score_match(self, row, mzs_c, ints_c, ppm_t):
        normalized_mf = self._normalize_sf(row.mf)
        if normalized_mf in self._target_db:
            return 5
        if normalized_mf in self._already_assigned:
            return 10
        return 1

    def _keep_list(self, row, mzs_c, ints_c, ppm_t):
        theor_spec = IsotopePattern("{}+{}".format(row.mf, row.adduct))
        theor_spec = theor_spec.centroids(resolutionAt(theor_spec.masses[0]))
        theor_spec = theor_spec.charged(1).trimmed(5)

        prev_intensity = None
        keep_list_all = np.ones_like(mzs_c, dtype=np.bool)
        for m, a in zip(theor_spec.masses, theor_spec.abundances):
            keep_list = 1e6 * np.abs(mzs_c - m) / m > ppm_t
            keep_list_all &= keep_list
            peak_list = np.where(np.logical_not(keep_list))[0]
            if len(peak_list) == 0:
                break
            intensity = ints_c[peak_list].sum()
            if prev_intensity and abs(intensity) > abs(prev_intensity):
                break
            prev_intensity = intensity

        return keep_list_all

    def fit_spectrum(self, mzs_c, ppms, ints_c, detection_limit, max_peaks=500):
        order = np.argsort(np.abs(ints_c))
        mzs_c = np.asarray(mzs_c)[order]
        ppms = np.asarray(ppms)[order]
        ints_c = np.asarray(ints_c)[order]

        mzs_c = mzs_c[ints_c >= detection_limit]
        ppms = ppms[ints_c >= detection_limit]
        ints_c = ints_c[ints_c >= detection_limit]

        n_peaks_initial = len(mzs_c)

        sf_list = []

        while len(mzs_c) > 0 and ints_c[-1] >= detection_limit:
            mz = mzs_c[-1]
            v = ints_c[-1]
            ppm = ppms[-1]
            hits = search_mz_candidates_pfg(mz, ['H', 'Na', 'K'], ppm_limit=ppm)
            if len(hits) == 0:
                print 'no hits for {}'.format(mz)
                mzs_c = mzs_c[:-1]
                ints_c = ints_c[:-1]
                ppms = ppms[:-1]
                continue

            scores = [self._score_match(candidate, mzs_c, ints_c, ppm)
                      for candidate in hits.itertuples()]

            best_hit_idx = np.argmax(scores)
            best_hit = hits.iloc[best_hit_idx]

            keep_list = self._keep_list(best_hit, mzs_c, ints_c, ppm)
            if np.all(keep_list):
                print 'wtf'
                mzs_c = mzs_c[:-1]
                ints_c = ints_c[:-1]
                ppms = ppms[:-1]
                continue

            mzs_c = mzs_c[keep_list]
            ints_c = ints_c[keep_list]
            ppms = ppms[keep_list]

            sf_list.append([v, best_hit])
            self._already_assigned.add(self._normalize_sf(best_hit['mf']))

            if len(sf_list) > max_peaks:
                break

        stats = {}
        stats['n_peaks_explained'] = n_peaks_initial - len(mzs_c)
        stats['n_molecules'] = len(sf_list)
        return sf_list, stats

assigner = MoleculeAssigner(db)

spec_fit = []
db_percentage = []
for ii in range(H.shape[0]):
    print "coeff", ii
    coeff_spec = H[ii]

    # FIXME make parameters adjustable
    detection_limit = 1e-3
    spec_fit.append(assigner.fit_spectrum(mz_axis, nmf_ppms, coeff_spec, detection_limit))

    sum_formulas = set([str(pyisocalc.parseSumFormula(x[1]['mf'])) for x in spec_fit[-1][0]])
    if len(spec_fit[-1][0]) > 0:
        db_percentage.append(len(db & sum_formulas) / float(len(spec_fit[-1][0])))
        print "database percentage in component", ii+1, ":", db_percentage[-1]

print "median percentage of database formulas:", np.median(db_percentage)

layers['layers_list'] = OrderedDict()
for ii in range(len(spec_fit)):
    layers['layers_list'][ii] = {}
    layers['layers_list'][ii]['image'] = W[:, ii].reshape(shape)[1:, 1:]
    layers['layers_list'][ii]['sf_list'] = []
    for sf in spec_fit[ii][0]:
        sf_a = "{}+{}".format(sf[1]['mf'], sf[1]['adduct'])
        mult = [sf[0], ]
        layers['layers_list'][ii]['sf_list'].append({"sf_a": sf_a, "mult": mult})

# save the layers for later analysis
with open(output_filename, "w+") as f:
    pickle.dump(layers, f)
