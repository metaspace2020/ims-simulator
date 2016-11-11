#!/usr/bin/env python

from cpyMSpec import centroidize
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
import numpy as np
import cPickle

import argparse


class NoiseGenerator(object):
    def __init__(self, nmf_fn, layers_fn, imzml_fn, inflate_noise):
        self._imzml = ImzMLParser(imzml_fn)
        self._inflate_noise=inflate_noise
        with np.load(nmf_fn) as data:
            nx, ny = data['shape']
            self._W = data['W'].reshape((nx, ny, -1))
            self._H = data['H']
            self._mz_axis = data['mz_axis']
        self._norm_real = {}
        self._norm_simulated = {}
        self._norm_groundtruth = {}
        self._norm_noise = {}
        self._norm_diff = {}
        self._coords = {}
        for i, coords in enumerate(self._imzml.coordinates):
            self._coords[(coords[0], coords[1])] = i
        self._mz_bins = []
        for mz, ppm in self._mz_axis:
            self._mz_bins.append(mz * (1.0 + 1e-6 * ppm))
        self._noiseDistributions = self._getNoiseDistribution()


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

    def _norm(self, intensities):
        return np.linalg.norm(intensities)

    def _getSpectrum(self, x, y):
        return self._imzml.getspectrum(self._coords[(x, y)])

    def _getRealSpectrum(self, x, y):
        real_spectrum = self._getSpectrum(x, y)
        real_mzs, real_intensities = map(np.array, real_spectrum)
        min_mz, max_mz = self._mz_bins[0], self._mz_bins[-1]
        inside_range = (real_mzs >= min_mz) & (real_mzs <= max_mz)
        real_mzs = real_mzs[inside_range]
        real_intensities = real_intensities[inside_range]
        return real_mzs, real_intensities

    def _getBinnedSpectrum(self, real_mzs, real_intensities):
        bins = np.digitize(real_mzs, self._mz_bins)
        n_bins = len(self._mz_bins)
        binned_real_intensities = np.bincount(bins, real_intensities, n_bins)
        return bins, binned_real_intensities

    def _getNoiseOneSpectrum(self, x, y):
        real_mzs, real_intensities = self._getRealSpectrum(x, y)
        thold = min(real_intensities)
        bins, binned_real_intensities = self._getBinnedSpectrum(real_mzs, real_intensities)
        binned_approx_intensities = self._W[x, y, :].dot(self._H)
        noise = np.abs(binned_real_intensities - binned_approx_intensities)*(binned_approx_intensities>thold)
        noise_intensities = noise[bins] * self._inflate_noise
        noise_mzs = np.array(real_mzs)
        nnz = noise_intensities > thold
        return noise_mzs[nnz], noise_intensities[nnz]

    def _approxMax(self):
        i_max = []
        for ii in np.random.choice(len(self._imzml.coordinates), 10):
            _mz, _ints = map(np.array, self._imzml.getspectrum(ii))
            if _ints.shape[0] == 0:
                continue
            i_max.append(np.max(_ints))
        return np.max(i_max)

    def _getIntensityNoiseDistribution(self, n_sample=50, max_sample=10000):
        def _get_noise_intensity(self, n_sample, max_sample):
            all_noise = []
            all_ints = []
            for ii in np.random.choice(range(len(self._imzml.coordinates)), n_sample):
                x, y = self._imzml.coordinates[ii]
                real_mzs, real_intensities = self._getRealSpectrum(x, y)
                bins, binned_real_intensities = self._getBinnedSpectrum(real_mzs, real_intensities)
                binned_approx_intensities = self._W[x, y, :].dot(self._H)
                binned_approx_intensities[
                    binned_approx_intensities < np.min(real_intensities[real_intensities > 0])] = 0
                noise = np.abs(binned_real_intensities - binned_approx_intensities)
                all_noise.extend(noise)
                all_ints.extend(binned_approx_intensities)
                if len(all_noise) > max_sample:
                    break
            return all_ints, all_noise

        def _fit_model(x, y):
            from sklearn import linear_model
            x, y = np.asarray(x), np.asarray(y)
            notnull = np.asarray([all((_x > 0., _y > 0.)) for _x, _y in zip(x, y)])
            x, y = np.log(x[notnull]), np.log(y[notnull])
            assert (x.shape == y.shape)
            model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
            model_ransac.fit(x.reshape(len(x), 1), y)
            fit = model_ransac.predict(x.reshape(x.shape[0], 1))
            resid = y[model_ransac.inlier_mask_] - fit[model_ransac.inlier_mask_]
            resid_std = np.std(resid)
            return model_ransac, resid_std

        x, y = _get_noise_intensity(self, n_sample, max_sample)
        self._model, self._resid_std = _fit_model(x, y)

    def _getNoiseDistribution(self):
        _mz_axis = np.asarray([m[0] for m in self._mz_axis])
        self._getIntensityNoiseDistribution()
        self._mz_hist = np.zeros(len(_mz_axis)+1)
        self._intensity_axis = np.linspace(0, self._approxMax(), 5000)
        self._intensity_hist = np.zeros(len(self._intensity_axis)+1)
        _noise_count = []
        for i in np.random.choice(len(self._imzml.coordinates), len(self._imzml.coordinates)*.1, replace=False): #reduce number of spectra for debuggin
            coords = self._imzml.coordinates[i]
            noise_mzs, noise_intensities = self._getNoiseOneSpectrum(coords[0], coords[1])
            self._mz_hist[np.digitize(noise_mzs, _mz_axis)] += 1
            self._intensity_hist[np.digitize(noise_intensities, self._intensity_axis)] += 1
            _noise_count.append(len(noise_mzs))
        # Histogram number of peaks
        self._noise_count_hist, _noise_count_edges = np.histogram(_noise_count, bins=1000)
        self._noise_count_axis = _noise_count_edges[0:-1]
        self._noise_count_hist = self._noise_count_hist.astype(float)
        # Clip histogram to match axis length
        self._intensity_hist = self._intensity_hist[0:-1]
        self._mz_hist = self._mz_hist[0:-1]
        # Normalise histogram weights
        self._noise_count_hist /= self._noise_count_hist.sum()
        self._mz_hist /= self._mz_hist.sum()
        self._intensity_hist /= self._intensity_hist.sum()


    def _spectrumPlusNoise(self,intensities):
        notnull = intensities > 0
        x = np.log(intensities[notnull])
        err = self._model.predict(x.reshape(len(x), 1))
        err_var = np.random.randn(len(x)) * (self._resid_std ** 0.5)
        err += err_var
        intensities_noise = np.zeros(intensities.shape)
        intensities_noise[notnull] = x + np.exp(err) * np.sign(err_var)
        return intensities_noise

    def _getDistribtionFromHist(self, axis, hist, n, dtype=float):
        ixs = np.random.choice(range(axis.shape[0]), p=hist, size=n)
        ixs[ixs == 0] = 1 # deal with edge cases
        vals = []
        for ii, ix in enumerate(ixs):
            vals.append((axis[ix - 1] - axis[ix]) * np.random.rand(1)[0] + axis[ix])
        return np.asarray(vals, dtype=dtype)

    def generateNoise(self, x, y):
        n_peaks = self._getDistribtionFromHist(self._noise_count_axis, self._noise_count_hist, 1, dtype=int)
        noise_mzs = self._getDistribtionFromHist(np.asarray([m[0] for m in self._mz_axis]), self._mz_hist, n_peaks)
        noise_intensities = self._getDistribtionFromHist(self._intensity_axis, self._intensity_hist, n_peaks)
        noise_intensities = self._spectrumPlusNoise(noise_intensities)
        return noise_mzs, noise_intensities

    def addNoise(self, profile_spectrum, coords):
        spec = map(np.array, profile_spectrum)
        p = centroidize(*spec)
        mzs = np.array(p.masses)
        mult = spec[1].max() if len(spec[1]) > 0 else 1
        intensities = np.array(p.abundances) * mult
        x, y = coords[:2]
        real_mzs, real_intensities = self._getRealSpectrum(*coords)
        limit = np.min(real_intensities)
        noise_mzs, noise_intensities = self.generateNoise(*coords)
        self._norm_noise[(x, y)] = self._norm(noise_intensities[noise_intensities > limit])
        self._norm_real[(x,y)] = self._norm(real_intensities[real_intensities > limit])
        self._norm_groundtruth[(x, y)] = self._norm(intensities[intensities > limit])
        self._norm_simulated[(x, y)] = self._norm_noise[(x, y)] + self._norm_groundtruth[(x, y)]
        self._norm_diff[(x, y)] = abs(self._norm_simulated[(x, y)] - self._norm(real_intensities))
        mzs = np.concatenate([mzs, noise_mzs])
        intensities = np.concatenate([intensities, noise_intensities])

        detectable = np.where(intensities > limit)[0]
        mzs = mzs[detectable]
        intensities = intensities[detectable]

        order = mzs.argsort()
        return mzs[order], intensities[order]

    def saveStatistics(self, filename):
        def toRect(d):
            xs = [k[0] for k in d]
            ys = [k[1] for k in d]
            img = np.zeros((max(xs) + 1, max(ys) + 1))
            for k in d:
                img[k[0], k[1]] = d[k]
            return img

        with open(filename, "w+") as f:
            np.savez(f,
                     real=toRect(self._norm_real),
                     simulated=toRect(self._norm_simulated),
                     groundtruth=toRect(self._norm_groundtruth),
                     noise=toRect(self._norm_noise),
                     diff=toRect(self._norm_diff))

def writeNoisySpectra(imzml_fname, output_fname, ng):
    imzml_sim = ImzMLParser(imzml_fname)
    with ImzMLWriter(output_fname, mz_dtype=np.float32) as w:
        for i, coords in enumerate(imzml_sim.coordinates):
            noisy_mzs, noisy_intensities = ng.addNoise(imzml_sim.getspectrum(i), coords)
            w.addSpectrum(noisy_mzs, noisy_intensities, coords)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="add noise to a clean dataset")
    parser.add_argument('real', type=str, help="original centroided imzML")
    parser.add_argument('simclean', type=str, help="input file produced by simulateClean.py")
    parser.add_argument('nmf', type=str, help="factorization file produced by NNMF.py")
    parser.add_argument('layers', type=str, help="layers file produces by assignMolecules.py")
    parser.add_argument('output', type=str, help="output filename (centroided .imzML)")
    parser.add_argument('--inflate-noise', type=float, default=1.0, help="noise inflation")

    args = parser.parse_args()
    print 'initialising noise generator'
    ng = NoiseGenerator(args.nmf, args.layers, args.real, args.inflate_noise)
    print 'generating noisy spectra'
    writeNoisySpectra(args.simclean, args.output, ng)
    print 'calculating stats'
    ng.saveStatistics(args.output + ".norms")