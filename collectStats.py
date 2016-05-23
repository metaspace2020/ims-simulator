#!/usr/bin/env python

from pyimzml.ImzMLParser import ImzMLParser
import numpy as np

def spectralHistograms(imzml):
    """
    Returns a dictionary:

    1) 'sparsity':
    Histogram of m/z differences between neighboring centroids in each spectrum.

    2) 'intensity':
    Histogram of centroid intensities
    """
    sparsity_hist_bins = np.linspace(-4, 1, 250)
    sparsity_hist = np.zeros(sparsity_hist_bins.shape[0] - 1, dtype=int)

    intensity_hist_bins = np.linspace(0, 10, 250)
    intensity_hist = np.zeros(intensity_hist_bins.shape[0] - 1, dtype=int)

    for i, coords in enumerate(imzml.coordinates):
        mzs, intensities = imzml.getspectrum(i)
        sparsity_hist += np.histogram(np.log10(np.diff(mzs)),
                                      sparsity_hist_bins)[0]

        intensity_hist += np.histogram(np.log10(intensities),
                                       intensity_hist_bins)[0]

    return {
        'sparsity': [sparsity_hist, sparsity_hist_bins],
        'intensity': [intensity_hist, intensity_hist_bins]
    }

def plotHistograms(stats_real, stats_sim, key):
    """
    Plot statistic distribution for real and simulated data on the same plot.
    Key must be one of 'sparsityHist', 'intensityHist'.
    First two arguments are the script-produced stats loaded via np.load
    """
    xlabels = {
        'sparsityHist': "log10(m/z difference between neighboring peaks)",
        'intensityHist': 'log10(centroid intensity)'
    }

    import matplotlib.pyplot as plt

    h0 = stats_real[key]
    h1 = stats_sim[key]

    plt.figure(figsize=(12, 6))

    mzs = h0[1][:-1]
    plt.fill_between(mzs, h0[0], color='b', alpha=0.5, label='Real')
    plt.fill_between(mzs, h1[0], color='g', alpha=0.5, label='Simulated')
    plt.xlabel(xlabels[key])
    plt.ylabel("Peak counts")
    plt.legend()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="collect statistics about IMS data")
    parser.add_argument('input', type=str, help="input file in .imzML format")
    parser.add_argument('output', type=str, help="output file (numpy-readable)")

    args = parser.parse_args()

    imzml = ImzMLParser(args.input)

    histograms = spectralHistograms(imzml)
    sparsityHist = histograms['sparsity']
    intensityHist = histograms['intensity']

    with open(args.output, "w+") as f:
        np.savez_compressed(f,
                            sparsityHist=sparsityHist,
                            intensityHist=intensityHist)
