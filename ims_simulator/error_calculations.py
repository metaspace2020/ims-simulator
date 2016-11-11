from rebin_dataset import do_rebinning, get_mz_range, open_dataset
import numpy as np
import os
import subprocess

def matrix_norms(dataset1, dataset2):
    # assumes same binning axis is used
    if not dataset1[0].shape == dataset2[0].shape:
        raise ValueError('datasets should be the same size {} {}'.format(dataset1[0].shape, dataset2[0].shape))
    residual = dataset1[0] - dataset2[0]
    print 'calculating norm'
    norm = residual.vnorm()
    print 'calcualting sums'
    sums = (residual.sum(axis=0).compute(), residual.sum(axis=1).compute())
    stats = {
        'residual_norm': norm,
        'sums': sums
    }
    print stats
    return stats

def pca_diff(dataset1, dataset2):
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=5, batch_size=10)
    d1_ipca = ipca.fit_transform(dataset1)
    d2_ipca = ipca.fit_transform(dataset2)
    err = np.abs(np.abs(d1_ipca) - np.abs(d2_ipca)).mean()
    stats= {
        'pca_abs_err':err
    }
    return stats

def compare_rebinned_datasets(dataset1, dataset2):
    stats = {}
    print 'matrix norms'
    stats.update(matrix_norms(dataset1, dataset2))
    print 'pca diff'
    stats.update(pca_diff(dataset1, dataset2))
    return stats

def compare_datasets(imzml_1, imzml_2, instrument_type = 'fticr', res200 = '200000'):
    for f_name in [imzml_1, imzml_2]:
        if not os.path.exists(f_name):
            raise IOError('file does not exist {}'.format(imzml_1))
    imzb_names = [os.path.splitext(imzml_1)[0] +'.imzb',
                  os.path.splitext(imzml_2)[0] + '.imzb',
                ]
    mz_range = [1e10, 0]
    for imzb_name, imzml_name in zip(imzb_names, [imzml_1, imzml_2]):
        if not os.path.exists(imzb_name):
            call_str = "ims"
            arg_str =  "convert"
            subprocess.check_call(call_str, arg_str, imzml_name, imzb_name)
        imzb = open_dataset(imzb_name)
        _mz_range = get_mz_range(imzb)
        mz_range[0] = np.min([mz_range[0], _mz_range[0]])
        mz_range[1] = np.max([mz_range[1], _mz_range[1]])
    stats = compare_rebinned_datasets(do_rebinning(imzb_names[0], instrument_type, float(res200), mz_range=mz_range), do_rebinning(imzb_names[1], instrument_type, float(res200), mz_range=mz_range))
    return stats


def save_results(results_dict, output_filename):
    with open(output_filename, "w+") as f:
        np.savez_compressed(f,
                           results=results_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="compare two IMS datasets")
    parser.add_argument('input_fullpath_1', type=str, help="input file1 in .imzML format")
    parser.add_argument('input_fullpath_2', type=str, help="input file2 in .imzML format")
    parser.add_argument('output', type=str, help="output file (numpy-readable)")

    args = parser.parse_args()
    stats = compare_datasets(args.input_fullpath_1, args.input_fullpath_2)
    save_results(stats, args.output)
