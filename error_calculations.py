from pyimzml.ImzMLParser import ImzMLParser
from rebin_dataset import do_rebinning
import numpy as np
import os
import subprocess

def matrix_norms(dataset1, dataset2):
    # assumes same binning axis is used
    if not dataset1.shape == dataset2.shape:
        raise ValueError('datasets should be the same size')
    residual = dataset1 - dataset2

    sums = (np.sum(residual, axis=0), np.sum(residual, axis=1))
    residual_bool = residual > 0

    hist = (np.sum(residual_bool, axis=0), np.sum(residual_bool, axis=1))
    stats = {
        'residual_norm': np.linalg.norm(residual),
        'hist': hist,
        'sums': sums
    }
    return stats

def compare_rebinned_datasets(dataset1, dataset2):
    stats = {}
    stats.update(matrix_norms(dataset1, dataset2))
    return stats

def compare_datasets(imzml_1, imzml_2, instrument_type = 'fticr', res200 = '200000'):
    imzb_names = [os.path.splitext(imzml_1.filename)[0]+'.imzb',
                  os.path.splitext(imzml_2.filename)[0] + '.imzb',
                ]
    for imzb_name, imzml_name in zip(imzb_names, [imzml_1, imzml_2]):
        if not os.path.exists(imzb_name):
            call_str = "ims convert {} {}".format(imzml_name, imzb_name)
            subprocess.check_call(call_str)

    stats = compare_rebinned_datasets(do_rebinning(imzb_names[0], instrument_type, res200), do_rebinning(imzb_names[1], instrument_type, res200))
    return stats


def save_results(results_dict, output_filename):
    with open(args.output, "w+") as f:
        np.savez_compressed(f,
                           results=results_dict)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="compare two IMS datasets")
    parser.add_argument('input_fullpath_1', type=str, help="input file1 in .imzML format")
    parser.add_argument('input_fullpath_2', type=str, help="input file2 in .imzML format")
    parser.add_argument('output', type=str, help="output file (numpy-readable)")

    args = parser.parse_args()

    imzml_1 = ImzMLParser(args.input_fullpath_1)
    imzml_2 = ImzMLParser(args.input_fullpath_2)

    stats = compare_datasets(imzml_1, imzml_2)
    save_results(stats, args.output)
