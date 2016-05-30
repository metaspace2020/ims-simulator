import numpy as np
from cpyMSpec import IsotopePattern
from cpyImagingMSpec import ImzbReader

import matplotlib.pyplot as plt

def img_correlation(i1, i2):
    i1 = np.sqrt(i1[i1 >= 0])
    i2 = np.sqrt(i2[i2 >= 0])
    return (i1 * i2).sum() / np.linalg.norm(i1) / np.linalg.norm(i2)

class NoiseCutoffEstimator(object):
    def __init__(self, imzb_filename):
        self.imzb = ImzbReader(imzb_filename)
        self.good_hits = None

    def findGoodHits(self, db, adducts=["+H", "+K", "+Na"], charge=1, ppm=5):
        mask = None
        good_hits = []
        corrs = []
        for sf in db:
            for adduct in adducts:
                p = IsotopePattern(sf + adduct).charged(charge)
                img1 = self.imzb.get_mz_image(p.masses[0], ppm)
                img2 = self.imzb.get_mz_image(p.masses[1], ppm)
                if mask is None:
                    mask = img1 >= 0

                img1 = img1[mask]
                img2 = img2[mask]
                if img2.sum() * img1.sum() == 0:
                    continue
                corr = img_correlation(img1, img2)
                if corr == 0:
                    continue
                corrs.append(corr)
                if corr >= 0.80 and (img2 > 0).sum() >= 0.5 * img2.shape[0]:
                    good_hits.append(sf + adduct)

        self.corrs = corrs
        self.charge = charge
        self.ppm = ppm
        self.mask = mask
        self.good_hits = good_hits

    def _flatSortedImages(self, sf):
        mzs = IsotopePattern(sf).charged(self.charge).masses
        imgs = []
        for i in range(2):
            img = self.imzb.get_mz_image(mzs[i], 5)
            imgs.append(img)

        i1 = imgs[0].ravel()
        i2 = imgs[1].ravel()
        i1 = np.sqrt(i1[i1 >= 0])
        i2 = np.sqrt(i2[i2 >= 0])
        order = i1.argsort()
        i1 = i1[order]
        i2 = i2[order]
        return i1, i2

    def _correlationPlot(self, i1, i2):
        scalar_products = np.cumsum(i1 * i2)
        norms1 = np.sqrt(np.cumsum(i1**2))
        norms2 = np.sqrt(np.cumsum(i2**2))

        nnz = (norms1 * norms2) > 0
        correlations = scalar_products[nnz] / (norms1[nnz] * norms2[nnz])

        return i1[nnz]**2, correlations

    def plotCorrelations(self, sf, *args, **kwargs):
        ints, corrs = self._correlationPlot(*self._flatSortedImages(sf))
        plt.plot(ints, corrs, *args, **kwargs)

    def estimateNoiseCutoff(self, min_corr=0.5):
        if self.good_hits is None:
            raise "Run findGoodHits() first!"

        noise_cutoffs = []
        mzs = []
        for i, hit in enumerate(self.good_hits):
            i1, i2 = self._flatSortedImages(hit)

            intensities, correlations = self._correlationPlot(i1, i2)

            low_corr_region = np.where(correlations < min_corr)[0]
            if len(low_corr_region) > 0:
                min_good_pos = np.max(low_corr_region) + 1
            else:
                min_good_pos = 0
            noise_cutoffs.append(intensities[min_good_pos])
            mzs.append(IsotopePattern(hit).charged(self.charge).masses[0])

        self.noise_cutoffs = noise_cutoffs
        self.noise_cutoff_mzs = mzs
        return np.median(self.noise_cutoffs)

    def plotImages(self, sf, n=3):
        p = IsotopePattern(sf).charged(self.charge)
        n = min(n, len(p.masses))
        imgs = []
        for i in range(n):
            plt.subplot(1, n, i + 1)
            img = self.imzb.get_mz_image(p.masses[i], self.ppm)
            imgs.append(img.copy())
            if (img > 0).sum() > 0:
                perc = np.percentile(img[img > 0], 99)
                img[img > perc] = perc
            img[img < 0] = 0
            plt.imshow(img)

        print "mz:", p.masses[0]
        print "median intensities:", [np.median(img[img > 0]) for img in imgs]
        print (imgs[1] > 0).sum(), "non-zero pixels in the second image"
        imgs[0] = imgs[0][self.mask]
        imgs[1] = imgs[1][self.mask]
        print "1st and 2nd isotopic image correlation:", img_correlation(imgs[0], imgs[1])
