import luigi

import pandas as pd
import yaml  # for parsing config file

from os.path import splitext, basename, expanduser
import subprocess
import tempfile
import shutil
import sys
import os
import pickle

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.cm import viridis as cmap
from matplotlib_venn import venn3
from sklearn.neighbors import NearestNeighbors

DECOY_ADDUCTS = ("+He,+Li,+Be,+B,+C,+N,+O,+F,+Ne,+Mg,+Al,+Si,+P,"
                 "+S,+Cl,+Ar,+Ca,+Sc,+Ti,+V,+Cr,+Mn,+Fe,+Co,+Ni,"
                 "+Cu,+Zn,+Ga,+Ge,+As,+Se,+Br,+Kr,+Rb,+Sr,+Y,+Zr,"
                 "+Nb,+Mo,+Ru,+Rh,+Pd,+Ag,+Cd,+In,+Sn,+Sb,+Te,+I,"
                 "+Xe,+Cs,+Ba,+La,+Ce,+Pr,+Nd,+Sm,+Eu,+Gd,+Tb,+Dy,"
                 "+Ho,+Ir,+Th,+Pt,+Os,+Yb,+Lu,+Bi,+Pb,+Re,+Tl,+Tm,"
                 "+U,+W,+Au,+Er,+Hf,+Hg,+Ta").split(",")

class AdductListParameter(luigi.Parameter):
    def parse(self, arguments):
        return tuple(arguments.split(','))

def unfreeze(x):
    if type(x) == luigi.parameter.FrozenOrderedDict:
        return {k: unfreeze(x[k]) for k in x}
    elif type(x) == tuple:
        return tuple([unfreeze(k) for k in x])
    elif type(x) == list:
        return [unfreeze(k) for k in x]
    else:
        return x

def get_id(obj):
    # horrible hacks to get a hash for luigi.DictParameter
    if type(obj) == luigi.parameter.FrozenOrderedDict:
        obj = unfreeze(obj)

    import json
    return str(hash(json.dumps(obj)) % 100000000)

def get_prefix(path):
    return splitext(basename(path))[0]

class CmdlineTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget(self.output_filename())

    def output_filename(self):
        raise NotImplementedError

    def program_args(self):
        raise NotImplementedError

    def output_is_stdout(self):
        return False

    def internal_script(self, name):
        return os.path.join(os.path.dirname(__file__), name)

    def before_run(self):
        pass

    def after_run(self):
        pass

    def run(self):
        self.before_run()
        args = [str(arg) for arg in self.program_args()]

        if self.output_is_stdout():
            f = tempfile.NamedTemporaryFile(delete=False)
            subprocess.check_call(args, stdout=f)
            f.close()
            shutil.move(f.name, self.output_filename())
        else:
            subprocess.check_call(args)
        self.after_run()

class CreateIsotopeDB(CmdlineTask):
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()
    groundtruth_fn = luigi.DictParameter('')

    def output_filename(self):
        db_name = get_prefix(self.molecular_db['sum_formulas_fn'])
        fn = "{}_{}_{}_{}.db".format(db_name,
                                     self.instrument['type'],
                                     self.instrument['res200'],
                                     get_id(self.molecular_db))
        if self.molecular_db['is_decoy']:
            fn += ".decoy"
        if self.groundtruth_fn:
            fn += ".gro" + get_id(self.groundtruth_fn)
        return fn

    def before_run(self):
        self._sum_formulas_fn = self.molecular_db['sum_formulas_fn']
        if self.groundtruth_fn and not self.molecular_db['is_decoy']:
            all_sfs = set()
            with open(self.molecular_db['sum_formulas_fn']) as in1:
                for l in in1:
                    all_sfs.add(l.strip())
            with open(self.groundtruth_fn) as in2:
                for l in in2:
                    all_sfs.add(l.split(',')[0])

            self._sum_formulas_fn += ".gro" + get_id(self.groundtruth_fn)
            with open(self._sum_formulas_fn, "w+") as out:
                for sf in all_sfs:
                    out.write("{}\n".format(sf))

    def program_args(self):
        return ['ims', 'isocalc',
                '--instrument', self.instrument['type'],
                '--resolution', self.instrument['res200'],
                '--adducts', ",".join(self.molecular_db['adducts']),
                self._sum_formulas_fn,
                self.output_filename()]

class ConvertImzMLToImzb(CmdlineTask):
    imzml_fn = luigi.Parameter()

    def output_filename(self):
        return get_prefix(self.imzml_fn) + ".imzb"

    def program_args(self):
        return ['ims', 'convert',
                self.imzml_fn, self.output_filename()]

class RunAnnotation(CmdlineTask):
    imzml_fn = luigi.Parameter()
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()
    groundtruth_fn = luigi.Parameter('')

    def requires(self):
        return {
            'imzb': ConvertImzMLToImzb(self.imzml_fn),
            'isotope_db': CreateIsotopeDB(self.instrument, self.molecular_db, self.groundtruth_fn),
        }

    def output_filename(self):
        db_id = get_id(self.molecular_db) + "."
        db_id += 'decoy' if self.molecular_db['is_decoy'] else 'target'
        if self.groundtruth_fn:
            db_id += "_gro" + get_id(self.groundtruth_fn)
        return get_prefix(self.imzml_fn) + ".results.db" + db_id + "_ins" + get_id(self.instrument)

    def output_is_stdout(self):
        return True

    def program_args(self):
        return ['ims', 'detect',
                self.input()['isotope_db'].fn,
                self.input()['imzb'].fn]

class GetAnnotationsForAdduct(luigi.Task):
    imzml_fn = luigi.Parameter()
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()
    adduct = luigi.Parameter()  # empty means annotations for all adducts
    groundtruth_fn = luigi.Parameter()

    def requires(self):
        return RunAnnotation(self.imzml_fn, self.instrument, self.molecular_db,
                             self.groundtruth_fn)

    def _output_filename(self):
        if self.adduct:
            return self.input().fn + "_" + self.adduct
        else:
            return self.input().fn

    def output(self):
        return luigi.LocalTarget(self._output_filename())

    def run(self):
        if self.adduct:
            df = pd.read_csv(self.input().fn).fillna(0)
            df[df['adduct'] == self.adduct].to_csv(self._output_filename(), index=False)
        else:
            return self.input()

class ComputeFDR(CmdlineTask):
    imzml_fn = luigi.Parameter()
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()
    adduct = luigi.Parameter('')
    groundtruth_fn = luigi.Parameter('')

    def requires(self):
        return {
            'target_results': GetAnnotationsForAdduct(self.imzml_fn, self.instrument, self.molecular_db, self.adduct, ''),  # , self.groundtruth_fn),
            'decoy_results': RunAnnotation(self.imzml_fn, self.instrument, self._decoy_molecular_db())
        }

    def output_filename(self):
        prefix = splitext(self.input()['decoy_results'].fn)[0] + "_" + get_id(self.molecular_db)
        if self.adduct:
            return prefix + "_" + self.adduct + ".fdr"
        else:
            return prefix + ".fdr"

    def _decoy_adducts(self):
        result = set()
        result.update(DECOY_ADDUCTS)

        for adduct in self.molecular_db['adducts']:
            if adduct in DECOY_ADDUCTS:
                result.remove(adduct)

        return tuple(result)

    def _decoy_molecular_db(self):
        return luigi.parameter.FrozenOrderedDict(
            sum_formulas_fn=self.molecular_db['sum_formulas_fn'],
            adducts=self._decoy_adducts(),
            is_decoy=True
        )

    def output_is_stdout(self):
        return True

    def program_args(self):
        args = ['ims', 'fdr',
                self.input()['target_results'].fn,
                self.input()['decoy_results'].fn]
        if self.groundtruth_fn:
            args += ['--groundtruth', self.groundtruth_fn]
        return args

class SimulationTask(CmdlineTask):
    config = luigi.DictParameter()

    @property
    def imzml_fn(self):
        return self.config['imzml']

    @property
    def instrument(self):
        return self.config['instrument']

    @property
    def molecular_db(self):
        return self.config['annotation']['database']

    def depends_on(self):
        return [k for k in self.config.keys() if k != 'imzml']

    def file_extension(self):
        raise NotImplementedError

    def output_filename(self):
        fn = get_prefix(self.imzml_fn)
        for key in self.depends_on():
            fn += "_" + key[:3] + get_id(self.config[key])
        return fn + "." + self.file_extension()

class ComputeFactorization(SimulationTask):
    def requires(self):
        return ConvertImzMLToImzb(self.imzml_fn)

    def depends_on(self):
        return ['instrument', 'factorization']

    def file_extension(self):
        return "nmf"

    def program_args(self):
        return [self.internal_script("NNMF.py"),
                self.input().fn, self.output_filename(),
                "--instrument", self.instrument['type'],
                "--res200", self.instrument['res200'],
                "--rank", self.config['factorization']['rank']]

class AssignMolecules(SimulationTask):
    def requires(self):
        return ComputeFactorization(self.config)

    def depends_on(self):
        return ['instrument', 'annotation', 'factorization']

    def file_extension(self):
        return "layers"

    def program_args(self):
        return [self.internal_script("assignMolecules.py"),
                self.input().fn, self.output_filename(),
                "--instrument", self.instrument['type'],
                "--res200", self.instrument['res200'],
                "--dynrange", self.config['annotation'].get('dynrange', 1000),
                "--db", self.molecular_db['sum_formulas_fn']]

    def after_run(self):
        with open(self.output_filename(), "r") as f:
            layers = pickle.load(f)
        if 'extra' in self.config['annotation']:
            for ion in self.config['annotation']['extra']:
                layers['layers_list'][ion['component']]['sf_list'].append(
                    {'sf_a': ion['sf'] + ion['adduct'],
                     'mult': ion['intensity'] / 100.0})
        with open(self.output_filename(), "w+") as f:
            pickle.dump(layers, f)

class ComputeStatistics(SimulationTask):
    def file_extension(self):
        return "stats"

    def depends_on(self):
        return []

    def program_args(self):
        return [self.internal_script("collectStats.py"),
                self.imzml_fn, self.output_filename()]

class SimulateCleanDataset(SimulationTask):
    def requires(self):
        return AssignMolecules(self.config)

    def file_extension(self):
        return "sim_profile_clean.imzML"

    def depends_on(self):
        return ['instrument', 'annotation', 'factorization']

    def program_args(self):
        return [self.internal_script("simulateClean.py"),
                self.input().fn, self.output_filename(),
                "--instrument", self.instrument['type'],
                "--res200", self.instrument['res200']]

class SimulateNoisyDataset(SimulationTask):
    def requires(self):
        return [SimulateCleanDataset(self.config),
                ComputeFactorization(self.config),
                GenerateGroundtruth(self.config)]

    def file_extension(self):
        return "sim.imzML"

    def program_args(self):
        return [self.internal_script("addNoise.py"),
                self.imzml_fn,
                self.input()[0].fn,
                self.input()[1].fn,
                self.input()[2].fn,
                self.output_filename(),
                "--inflate-noise", self.config['noise']['inflation']]

class GenerateGroundtruth(SimulationTask):
    def file_extension(self):
        return "groundtruth"

    def depends_on(self):
        return ['instrument', 'annotation', 'factorization']

    def requires(self):
        return AssignMolecules(self.config)

    def program_args(self):
        return [self.internal_script("layersToGroundtruth.py"),
                self.input().fn,
                self.output_filename()]

def simulatedDataFilename(config):
    return SimulateNoisyDataset(config).output_filename()

def simulatedDataConfig(config):
    return luigi.parameter.FrozenOrderedDict(
        imzml=simulatedDataFilename(config),
        annotation=config['annotation'],
        instrument=config['instrument'],
        groundtruth=GenerateGroundtruth(config).output_filename(),
        factorization=config['factorization'],
        noise=config['noise']
    )

class ComputeBasicStats(SimulationTask):
    def depends_on(self):
        return ['imzml']

    def file_extension(self):
        return "stats"

    def program_args(self):
        return [self.internal_script("collectStats.py"),
                self.config['imzml'],
                self.output_filename()]

class RunFullPipeline(SimulationTask):
    def requires(self):
        imzml = self.config['imzml']
        instrument = self.config['instrument']
        db = self.config['annotation']['database']
        if 'groundtruth' in self.config:
            groundtruth_fn = self.config['groundtruth']
        else:
            groundtruth_fn = None

        return {
            'fdr': {
                adduct: ComputeFDR(imzml, instrument, db, adduct, groundtruth_fn)
                for adduct in db['adducts']
            },
            'stats': ComputeBasicStats(self.config),
            'nnmf': ComputeFactorization(self.config)
        }

    def file_extension(self):
        return "output_locations"

    def run(self):
        paths = {
            'stats': self.input()['stats'].fn,
            'nnmf': self.input()['nnmf'].fn,
            'imzml': self.config['imzml']
        }

        fdr = self.input()['fdr']
        paths['fdr'] = {a: fdr[a].fn for a in fdr}

        with open(self.output_filename(), "w+") as f:
            yaml.dump(paths, f)

class CreateAnnotationConfigForSimulatedData(luigi.Task):
    config = luigi.DictParameter()
    priority = 10

    def requires(self):
        return SimulateNoisyDataset(self.config)

    def output(self):
        return luigi.LocalTarget("simulated_{}_config.yaml".format(get_id(self.config)))

    def run(self):
        with self.output().open("w") as f:
            simulation_config = unfreeze(simulatedDataConfig(self.config))
            yaml.dump(simulation_config, f)

def generateReport(orig_yaml_fn, sim_yaml_fn, groundtruth_fn, output_filename):
    plt.ioff()

    orig = yaml.load(open(orig_yaml_fn))
    sim = yaml.load(open(sim_yaml_fn))

    def loadStats(path):
        result = {}
        with open(path) as fin:
            with np.load(fin) as d:
                result['sparsityHist'] = d['sparsityHist']
                result['intensityHist'] = d['intensityHist']
        return result

    def loadNNMF(fn):
        with open(fn) as f:
            with np.load(f) as d:
                return {'W': d['W'], 'H': d['H'], 'shape': d['shape'],
                        'mz_axis': np.array([x[0] for x in d['mz_axis']])}

    orig_stats = loadStats(orig['stats'])
    sim_stats = loadStats(sim['stats'])

    orig_nnmf = loadNNMF(orig['nnmf'])
    sim_nnmf = loadNNMF(sim['nnmf'])

    bin_size = 0.01

    def getImageComponents(nnmf):
        return nnmf['W'].T

    common_mz_axis = np.arange(min(min(orig_nnmf['mz_axis']),
                                   min(sim_nnmf['mz_axis'])),
                               max(max(orig_nnmf['mz_axis']),
                                   max(sim_nnmf['mz_axis'])),
                               bin_size)

    def getSpectralComponents(nnmf):
        axis_len = len(common_mz_axis)
        result = np.zeros((len(nnmf['H']), axis_len))
        indices = np.digitize(nnmf['mz_axis'], common_mz_axis)
        for i, h in enumerate(nnmf['H']):
            result[i, :] = np.bincount(indices, h, minlength=axis_len)[:axis_len]
        return result

    def plotImageComponent(nnmf, i):
        plt.imshow(nnmf['W'][:, i].reshape(nnmf['shape']), cmap=cmap)
        plt.colorbar()

    def plotSpectralComponent(nnmf, i):
        plt.plot(nnmf['mz_axis'], nnmf['H'][i])

    def plotSparsityHistogram(stats, real=True):
        count, logdist = stats['sparsityHist'][:2]
        plt.plot(logdist[:-1], count)
        plt.title("Sparsity histogram for {} data".format('real' if real else 'simulated'))
        plt.xlabel("log10(distance between adjacent centroids)")
        plt.ylabel("peak count")

    def plotIntensityHistogram(stats, real=True):
        count, logint = stats['intensityHist'][:2]
        plt.plot(logint[:-1], count)
        plt.title("Intensity histogram for {} data".format('real' if real else 'simulated'))
        plt.xlabel("log10(peak intensity)")
        plt.ylabel("peak count")

    def plotComponentSimilarity(factor):
        nn1 = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine')
        orig_factor = factor(orig_nnmf)
        sim_factor = factor(sim_nnmf)
        nn1 = nn1.fit(orig_factor)
        distances, indices = nn1.kneighbors(sim_factor)
        dim = len(orig_factor)
        heatmap = np.zeros([dim] * 2)
        heatmap[:] = np.nan
        for i, (d, j) in enumerate(zip(distances, indices)):
            heatmap[i, j] = d
        cmap.set_bad('w', 1.)
        plt.pcolor(np.ma.array(heatmap, mask=np.isnan(heatmap)),
                   cmap=cmap, vmin=0, vmax=1)
        plt.colorbar()

    def plotVennDiagram(fdr_threshold):
        groundtruth = pd.read_csv(groundtruth_fn, names=['formula', 'adduct'])
        sim_layer_formulas = groundtruth.groupby('adduct')

        def top_results(df, threshold, adduct):
            """
            results with estimated FDR < threshold and positive MSM
            """
            fdr = df['fdr'] if 'fdr' in df else df['est_fdr']
            return df[(fdr < threshold) & (df['adduct'] == adduct) &
                      (df['img'] * df['iso'] * df['moc'] > 0)]

        for i, adduct in enumerate(orig['fdr'].keys()):
            plt.subplot(len(orig['fdr']), 1, i + 1)
            plt.title("Annotation overlap for {} (FDR threshold = {})"
                      .format(adduct, fdr_threshold))
            orig_res = pd.read_csv(orig['fdr'][adduct])
            sim_res = pd.read_csv(sim['fdr'][adduct])
            db = set(orig_res['formula'])

            orig_top = set(top_results(orig_res, fdr_threshold, adduct)['formula'])
            sim_top = set(top_results(sim_res, fdr_threshold, adduct)['formula'])
            venn3([orig_top, sim_top,
                   set(sim_layer_formulas.get_group(adduct)['formula']) & db],
                  ("Orig. annotations", "Sim. annotations", "Sim. groundtruth & DB"))

    def createFigure():
        return plt.figure(figsize=(8.27, 11.69), dpi=100)  # A4 format

    def saveFigure(fig, pdf):
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(output_filename) as pdf:
        n_components = len(orig_nnmf['H'])
        per_page = 5
        for i in range(0, n_components / per_page + 1):
            n_plots = min(per_page, n_components - i * per_page)
            if n_plots < 1:
                break
            fig = createFigure()
            for j in range(n_plots):
                plt.subplot(n_plots, 2, 2 * j + 1)
                plotImageComponent(orig_nnmf, per_page * i + j)
                plt.subplot(n_plots, 2, 2 * j + 2)
                plt.xticks(rotation=45)
                plotSpectralComponent(orig_nnmf, per_page * i + j)
            saveFigure(fig, pdf)

        for plotFunc in [plotSparsityHistogram, plotIntensityHistogram]:
            fig = createFigure()
            plt.subplot(2, 1, 1)
            plotFunc(orig_stats, real=True)
            plt.subplot(2, 1, 2)
            plotFunc(sim_stats, real=False)
            saveFigure(fig, pdf)

        fig = createFigure()
        plt.subplot(2, 1, 1)
        plt.title("Cosine distances between image components")
        plt.xlabel("Components of original data")
        plt.ylabel("Components of simulated data")
        plotComponentSimilarity(getImageComponents)

        plt.subplot(2, 1, 2)
        plt.title("Cosine distances between spectral components (rebinned with step {})".format(bin_size))
        plt.xlabel("Components of original data")
        plt.ylabel("Components of simulated data")
        plotComponentSimilarity(getSpectralComponents)
        saveFigure(fig, pdf)

        fig = createFigure()
        plotVennDiagram(0.2)
        saveFigure(fig, pdf)

        def plotNormRatios(a, b):
            plt.imshow(a / b, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar()
        norms = np.load(open(sim['imzml'] + '.norms'))

        fig = createFigure()
        plt.subplot(3, 1, 1)
        plt.title("||real - simulated||_2 / ||real||_2")
        plotNormRatios(norms['diff'], norms['real'])
        plt.subplot(3, 1, 2)
        plt.title("||groundtruth||_2 / ||simulated||_2")
        plotNormRatios(norms['groundtruth'], norms['simulated'])
        plt.subplot(3, 1, 3)
        plt.title("||noise||_2 / ||simulated||_2")
        plotNormRatios(norms['noise'], norms['simulated'])
        saveFigure(fig, pdf)

class ComputeSimilarityMetrics(luigi.Task):
    config = luigi.DictParameter()

    def requires(self):
        return {'original': RunFullPipeline(self.config),
                'sim_config': CreateAnnotationConfigForSimulatedData(self.config),
                'groundtruth': GenerateGroundtruth(self.config),
                'simulated': RunFullPipeline(simulatedDataConfig(self.config))}

    def output(self):
        return luigi.LocalTarget("report_{}.pdf".format(get_id(self.config)))

    def run(self):
        generateReport(self.input()['original'].fn, self.input()['simulated'].fn,
                       self.input()['groundtruth'].fn, self.output().fn)

def _ordered_dict(loader, node):
    loader.flatten_mapping(node)
    return luigi.parameter.FrozenOrderedDict(loader.construct_pairs(node))

yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                     _ordered_dict)

def readConfig(filename):
    with open(filename) as conf:
        config = yaml.load(conf)

    cfg = config.get_wrapped()
    cfg['imzml'] = expanduser(config['imzml'])
    cfg['annotation']['database'].get_wrapped()['sum_formulas_fn'] = expanduser(config['annotation']['database']['sum_formulas'])
    cfg['annotation']['database'].get_wrapped()['adducts'] = tuple(cfg['annotation']['database']['adducts'])
    if 'extra' in cfg['annotation']:
        cfg['annotation'].get_wrapped()['extra'] = tuple(cfg['annotation']['extra'])
    cfg['annotation']['database'].get_wrapped()['is_decoy'] = False
    return config

if __name__ == '__main__':
    config = readConfig(sys.argv[1])
    print config
    luigi.build([ComputeSimilarityMetrics(config)], local_scheduler=True)
