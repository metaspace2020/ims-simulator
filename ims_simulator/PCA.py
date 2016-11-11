import argparse
import sys
from pyimzml import ImzMLParser
from mz_axis import generate_mz_axis
import numpy as np
from rebin_dataset import get_instrument_model

def get_binned_spectrum(imzml, i, mz_axis, min_length):
    mzs, ints = imzml.getspectrum(i)
    mzs = np.asarray(mzs)
    inds = np.digitize(mzs, mz_axis)
    intsBinned = np.bincount(inds, weights=mzs, minlength=min_length)
    return intsBinned

def random_projection_pixels_multiple_datasets(imzmls, mz_axis, rp_rank=[], rp_matrix=[]):
    # input checks/ tidy
    assert(not all([rp_rank==[], rp_matrix==[]]))
    if rp_matrix==[]:
        rp_matrix = np.random.rand(len(mz_axis)+1, rp_rank)
    else:
        assert(rp_matrix.shape[0] == len(mz_axis)+1)
        rp_rank = rp_matrix.shape[1]
    # main algorithm

    rp_projected = np.array([]).reshape(0, rp_rank)
    for imzml in imzmls:
        _rp_projected = np.zeros((len(imzml.coordinates), rp_rank))
        for i, xyz in enumerate(imzml.coordinates):
            if i%int(len(imzml.coordinates)/20) == 0:
                print np.round(100.*i/len(imzml.coordinates), decimals=1)
            min_length = len(mz_axis) + 1
            intsBinned = get_binned_spectrum(imzml, i, mz_axis, min_length)
            _rp_projected[i,:] = intsBinned.dot(rp_matrix)
        rp_projected = np.vstack([rp_projected, _rp_projected])
    return rp_projected


def basis_approximation_pca_mulitple_datasets(imzml_filenames, instrument_type, res200, rp_rank = 250):
    imzmls = [ImzMLParser.ImzMLParser(imzml_filename) for imzml_filename in imzml_filenames]
    mz_min = 100
    mz_max = 1000
    print 'generating mz axis'
    instrument = get_instrument_model(instrument_type, res200)
    mz_axis = generate_mz_axis(mz_min, mz_max, instrument=instrument, step_size=1)
    mz_axis = np.asarray(mz_axis)[:, 0]
    rp_projected = np.zeros((len(mz_axis)+1, rp_rank))
    for imzml in imzmls:
        rp_matrix = np.random.rand(len(imzml.coordinates), rp_rank)
        rp_projected += project_pixels(imzml, mz_axis, rp_matrix)
    basis = np.linalg.qr(rp_projected, mode="reduced")[0]
    projected = np.array([]).reshape(0, rp_rank)
    for imzml in imzmls:
        _projected = project_spectra(imzml, mz_axis, basis)

        projected = np.vstack([projected, _projected])
        projected -= projected.mean(axis=0)
        #projected /= projected.std(axis=0)
    U, s, V = np.linalg.svd(projected, full_matrices=0)
    return U, s, V


def basis_approximation(imzml, mz_axis, rp_rank):
    rp_projected = random_projection_pixels(imzml, mz_axis, rp_rank)
    basis = np.linalg.qr(rp_projected, mode="reduced")[0]
    projected = project_spectra(imzml, mz_axis, basis)
    return projected, basis


def basis_approximation_pca(imzml, mz_axis, rp_rank):
    print 'calculating basis'
    projected, basis = basis_approximation(imzml, mz_axis, rp_rank)
    print 'projecting data'
    projected -= projected.mean(axis=0)
    print 'computing pca'
    U, s, V = np.linalg.svd(projected, full_matrices=0)
    return U,s,V


def do_pca_multiple_datasets(imzml_filenames, instrument_type, res200, rank, args={}, rp_rank = 0):
    mz_min = 100
    mz_max = 1000
    print 'generating mz axis'
    instrument = get_instrument_model(instrument_type, res200)
    mz_axis = generate_mz_axis(mz_min, mz_max, instrument=instrument, step_size=1)
    mz_axis = np.asarray(mz_axis)[:, 0]
    rp_matrix = np.random.rand(len(mz_axis)+1, rp_rank)
    rp_projected = random_projection_pixels_multiple_datasets([ImzMLParser.ImzMLParser(imzml_filename) for imzml_filename in imzml_filenames],
                                                              mz_axis,
                                                              rp_matrix=rp_matrix)
    U, s, V = np.linalg.svd(rp_projected, full_matrices=0)
    return U, s, V


def project_spectra(imzml, mz_axis, basis):
    min_length = len(mz_axis) + 1
    projected = np.zeros((len(imzml.coordinates), basis.shape[1]))
    for i, xyz in enumerate(imzml.coordinates):
        if i % int(len(imzml.coordinates) / 20) == 0:
            print np.round(100. * i / len(imzml.coordinates), decimals=1)
        intsBinned = get_binned_spectrum(imzml, i, mz_axis, min_length)
        projected[i, :] = intsBinned.dot(basis)
    return projected


def project_pixels(imzml, mz_axis, basis):
    projected = np.zeros((len(mz_axis)+1, basis.shape[1]))
    min_length = len(mz_axis) + 1
    for i, xyz in enumerate(imzml.coordinates):
        if i % int(len(imzml.coordinates) / 20) == 0:
            print np.round(100. * i / len(imzml.coordinates), decimals=1)
        intsBinned = get_binned_spectrum(imzml, i, mz_axis, min_length)
        projected += np.outer(intsBinned, basis[i, :])
    return projected


def random_projection_spectra(imzml, mz_axis, rp_rank=[], rp_matrix=[]):
    if rp_matrix==[]:
        rp_matrix = np.random.rand(len(mz_axis)+1, rp_rank)
        assert not rp_rank==[]
    else:
        assert(rp_matrix.shape[0] == len(len(mz_axis)+1))
    rp_projected = project_spectra(imzml, mz_axis, rp_matrix)
    return rp_projected


def random_projection_pixels(imzml, mz_axis, rp_rank, rp_matrix=[]):
    if rp_matrix == []:
        rp_matrix = np.random.rand(len(imzml.coordinates), rp_rank)
        assert not rp_rank == []
    else:
        assert (rp_matrix.shape[0] == len(len(mz_axis) + 1))
    rp_projected = project_pixels(imzml, mz_axis, rp_matrix)
    return rp_projected


def do_pca_coeffs(imzml_filename, instrument_type, res200, rank, args={}, rp_rank = 0):
    imzml = ImzMLParser.ImzMLParser(imzml_filename)
    mz_min = 100
    mz_max = 1000
    print 'generating mz axis'
    instrument = get_instrument_model(instrument_type, res200)
    mz_axis = generate_mz_axis(mz_min, mz_max, instrument=instrument, step_size=1)
    mz_axis = np.asarray(mz_axis)[:,0]
    if rp_rank == 0:
        rp_rank = 5*rank
    print 'doing projection'
    rp_projected = random_projection_pixels(imzml, mz_axis, rp_rank)
    U, s, V = np.linalg.svd(rp_projected, full_matrices=0)
    return U, s, V

def do_pca(imzml_filename, instrument_type, res200, rank, args={}, rp_rank = 0):
    imzml = ImzMLParser.ImzMLParser(imzml_filename)
    mz_min = 100
    mz_max = 1000
    print 'generating mz axis'
    instrument = get_instrument_model(instrument_type, res200)
    mz_axis = generate_mz_axis(mz_min, mz_max, instrument=instrument, step_size=1)
    mz_axis = np.asarray(mz_axis)[:,0]
    if rp_rank == 0:
        rp_rank = 5*rank
    print 'doing projection'
    rp_projected = random_projection_spectra(imzml, mz_axis, rp_rank)
    U, s, V = np.linalg.svd(rp_projected, full_matrices=0)
    return U, s, V

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="compute NMF of a centroided dataset")
    parser.add_argument('input_file_imzml', type=str, help="input file in .imzml format")
    parser.add_argument('output_file_np', type=str, help="output file (numpy-readable NMF)")
    parser.add_argument('--instrument', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
    parser.add_argument('--res200', type=float, default=140000)
    parser.add_argument('--rank', type=int, default=10, help="desired factorization rank")
    args = parser.parse_args()
    if args.rank < 3:
        sys.stdout.write("Factorization rank must be at least 10! Exiting.\n")
        sys.exit(1)
    m = do_pca(args.input_file_imzml, args.output_file_np, args.instrument, float(args.res200), float(args.rank), args)
    np.savez_compressed(open(args.output_file_np), m)