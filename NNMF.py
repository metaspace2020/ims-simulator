#!/usr/bin/env python
# coding: utf-8

from cpyImagingMSpec import ImzbReader
import dask.array as da
import numpy as np
from toolz import partition_all

from external.nnls import nnlsm_blockpivot

import argparse

parser = argparse.ArgumentParser(description="compute NMF of a centroided dataset")
parser.add_argument('input', type=str, help="input file in .imzb format")
parser.add_argument('output', type=str, help="output file (numpy-readable NMF)")
parser.add_argument('--instrument', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
parser.add_argument('--res200', type=float, default=140000)
parser.add_argument('--rank', type=int, default=40, help="desired factorization rank")

args = parser.parse_args()
import sys
if args.rank < 10:
    sys.stdout.write("Factorization rank must be at least 10! Exiting.\n")
    sys.exit(1)

def resolutionAt(mz):
    if args.instrument == 'orbitrap':
        return args.res200 * (200.0 / mz) ** 0.5
    elif args.instrument == 'fticr':
        return args.res200 * (200.0 / mz)

imzb = ImzbReader(args.input)

def generate_mz_axis(mz_min, mz_max, step_size=5):
    """
    returns array of non-overlapping tuples (mz, ppm) that cover [mz_min, mz_max]
    """
    mz_axis = []
    mz = mz_min
    while mz < mz_max:
        fwhm = mz / resolutionAt(mz)
        step = step_size * fwhm
        ppm = 1e6 * step / (2.0 * mz + step)
        mz_axis.append((mz + step/2, ppm))
        mz += step
    return mz_axis

# FIXME: read m/z range from the input file
mz_axis = generate_mz_axis(100, 1000)
print "Number of m/z bins:", len(mz_axis)

def get_mz_images(mz_axis_chunk):
    imgs = np.zeros((len(mz_axis_chunk), imzb.height, imzb.width))
    for n, (mz, ppm) in enumerate(mz_axis_chunk):
        img = imzb.get_mz_image(mz, ppm)
        img[img < 0] = 0
        perc = np.percentile(img, 99)
        img[img > perc] = perc
        imgs[n, :, :] = img
    return imgs

K = 100

mz_axis_chunks = list(partition_all(K, mz_axis))

# create dask array manually using tasks
tasks = {('x', i, 0, 0): (get_mz_images, mz_chunk) for i, mz_chunk in enumerate(mz_axis_chunks)}

chunks_mz = [len(c) for c in mz_axis_chunks]
chunks_x = (imzb.height, )
chunks_y = (imzb.width, )
arr = da.Array(tasks, 'x', chunks=(chunks_mz, chunks_x, chunks_y), dtype=float)
print arr.shape

print "Computing bin intensities... (takes a while)"
image_intensities = arr.sum(axis=(1, 2)).compute()

N_bright = 500
bright_images_pos = image_intensities.argsort()[::-1][:N_bright]
mz_axis_pos = np.array(mz_axis)[bright_images_pos]
arr_pos = arr[bright_images_pos]
print "Selected top", N_bright, "brightest images for NNMF"
print arr_pos.shape

cols = []
r = args.rank

def nnls_frob(x, anchors):
    ncols = x.shape[1]
    x_sel = np.array(anchors)
    # print "projection"
    result = np.zeros((x_sel.shape[1], ncols))

    # apply NNLS to chunks so as to avoid loading all m/z images into RAM
    for chunk in partition_all(100, range(ncols)):
        residuals = np.array(x[:, chunk])
        result[:, chunk] = nnlsm_blockpivot(x_sel, residuals)[0]

    return result

# treat images as vectors (flatten them)
x = arr_pos.reshape((arr_pos.shape[0], -1)).T

print "Running non-negative matrix factorization"

# apply XRay algorithm
# ('Fast conical hull algorithms for near-separable non-negative matrix factorization' by Kumar et. al., 2012)
R = x
while len(cols) < r:
    # print "detection"
    p = da.random.random(x.shape[0], chunks=x.shape[0])
    scores = (R * x).sum(axis=0)
    scores /= p.T.dot(x)
    scores = np.array(scores)
    scores[cols] = -1
    best_col = np.argmax(scores)
    assert best_col not in cols
    cols.append(best_col)
    print "picked {}/{} columns".format(len(cols), r)

    H = nnls_frob(x, x[:, cols])
    R = x - da.dot(x[:, cols], da.from_array(H, H.shape))

    if len(cols) > 0 and len(cols) % 5 == 0:
        residual_error = da.vnorm(R, 'fro').compute() / da.vnorm(x, 'fro').compute()
        print "relative error is", residual_error

W = np.array(x[:, cols])

residual_error = da.vnorm(R, 'fro').compute() / da.vnorm(x, 'fro').compute()
print "Finished column picking, relative error is", residual_error

print "Projecting all m/z bin images on the obtained basis..."
H_full = nnls_frob(arr.reshape((arr.shape[0], -1)).T,
                   arr_pos.reshape((arr_pos.shape[0], -1))[cols, :].T)

print "Computing noise statistics..."
noise_stats = {'prob': [], 'sqrt_median': [], 'sqrt_std': []}
percent_complete = 5.0

min_intensities = np.zeros((imzb.height, imzb.width))
min_intensities[:] = np.inf

for i, (mz, ppm) in enumerate(mz_axis):
    orig_img = imzb.get_mz_image(mz, ppm)
    orig_img[orig_img < 0] = 0
    approx_img = W.dot(H_full[:, i]).reshape((imzb.height, imzb.width))
    diff = orig_img - approx_img
    noise = diff[diff > 0]

    mask = orig_img > 0
    min_intensities[mask] = np.minimum(min_intensities[mask], orig_img[mask])

    noise_prob = float(len(noise)) / (imzb.width * imzb.height)

    noise_stats['prob'].append(noise_prob)

    if noise_prob > 0:
        noise = np.sqrt(noise)
        noise_stats['sqrt_median'].append(np.median(noise))
        noise_stats['sqrt_std'].append(np.std(noise))
    else:
        noise_stats['sqrt_median'].append(0)
        noise_stats['sqrt_std'].append(0)
    if float(i + 1) / len(mz_axis) * 100.0 > percent_complete:
        print "{}% done".format(percent_complete)
        percent_complete += 5
print "100% done"

with open(args.output, "w+") as f:
    np.savez_compressed(f, W=W, H=H_full, mz_axis=mz_axis, shape=(imzb.height, imzb.width),
                        noise_prob=np.array(noise_stats['prob']),
                        noise_sqrt_avg=np.array(noise_stats['sqrt_median']),
                        noise_sqrt_std=np.array(noise_stats['sqrt_std']),
                        min_intensities=min_intensities)
    print "Saved NMF and noise stats to {} (use numpy.load to read it)".format(args.output)
