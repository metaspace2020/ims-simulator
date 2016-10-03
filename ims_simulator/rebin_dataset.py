from cpyImagingMSpec import ImzbReader
import argparse
import dask.array as da
import numpy as np
from toolz import partition_all
from mz_axis import generate_mz_axis, Instrument


def get_mz_images(mz_axis_chunk, imzb):
    imgs = np.zeros((len(mz_axis_chunk), imzb.height, imzb.width))
    for n, (mz, ppm) in enumerate(mz_axis_chunk):
        img = imzb.get_mz_image(mz, ppm)
        img[img < 0] = 0
        perc = np.percentile(img, 99)
        img[img > perc] = perc
        imgs[n, :, :] = img
    return imgs


def open_dataset(input_filename):
    imzb = ImzbReader(input_filename)
    return imzb


def get_mz_range(imzb):
    mz_range = (imzb.min_mz - 0.1, imzb.max_mz + 0.1)
    assert mz_range[0] > 0
    assert mz_range[1] > 1
    return mz_range


def get_mz_axis(min_mz, max_mz, instrument, step=1.0):
    mz_axis = generate_mz_axis(min_mz, max_mz, instrument, step)
    return mz_axis


def save_image_intensities(image_intensities, output_filename, instrument, mz_axis):
    with open(output_filename, "w+") as f:
        np.savez_compressed(f, image_intensities=image_intensities, instrument=instrument, mz_axis=mz_axis)


def parallel_rebin(K, mz_axis, imzb):
    mz_axis_chunks = list(partition_all(K, mz_axis))
    # create dask array manually using tasks
    tasks = {('x', i, 0, 0): (get_mz_images, mz_chunk, imzb) for i, mz_chunk in enumerate(mz_axis_chunks)}
    chunks_mz = [len(c) for c in mz_axis_chunks]
    chunks_x = (imzb.height,)
    chunks_y = (imzb.width,)
    arr = da.Array(tasks, 'x', chunks=(chunks_mz, chunks_x, chunks_y), dtype=float)
    print arr.shape
    return arr


def get_instrument_model(instrument_type, res200):
    class hackalot():
        def __init__(self, instrument_type, res200):
            self.res200=res200
            self.instrument = instrument_type
    foo = hackalot(instrument_type, res200)
    return Instrument(foo)


def do_rebinning(filename_imzb, instrument_type, res200, mz_range=[]):
    imzb = open_dataset(filename_imzb)
    if mz_range == []:
        mz_range = get_mz_range(imzb)
    K = 20 #in principle can be calcualted from system memory
    instrument = get_instrument_model(instrument_type, res200)
    mz_axis = get_mz_axis(mz_range[0], mz_range[1], instrument, 1.0)
    image_intensities = parallel_rebin(K, mz_axis, imzb)
    return image_intensities, mz_axis


def do_and_save_rebinning(input_file_imzb, output_file_np, instrument_type, res200):
    image_intensities, mz_axis = do_rebinning(input_file_imzb, instrument_type, res200)
    save_image_intensities(image_intensities, output_file_np, instrument_type, mz_axis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="rebin a centroided dataset")
    parser.add_argument('input_file_imzb', type=str, help="input file in .imzb format")
    parser.add_argument('output_file_np', type=str, help="output file (numpy-readable)")
    parser.add_argument('--instrument_type', type=str, default='orbitrap', choices=['orbitrap', 'fticr'])
    parser.add_argument('--res200', type=float, default=140000)
    args = parser.parse_args()
    print 'started'
    do_and_save_rebinning(args.input_file_imzb, args.output_file_np, args.instrument_type, args.res200)
