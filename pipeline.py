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
            'target_results': GetAnnotationsForAdduct(self.imzml_fn, self.instrument, self.molecular_db, self.adduct, self.groundtruth_fn),
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
        grountruth=GenerateGroundtruth(config).output_filename(),
        factorization=config['factorization'],
        noise=config['noise']
    )

class RunFullPipeline(SimulationTask):
    def requires(self):
        imzml = self.config['imzml']
        instrument = self.config['instrument']
        db = self.config['annotation']['database']
        if 'grountruth' in self.config:
            groundtruth_fn = self.config['grountruth']
        else:
            groundtruth_fn = None

        return {adduct: ComputeFDR(imzml, instrument,
                                   db, adduct, groundtruth_fn)
                for adduct in db['adducts']}

    def file_extension(self):
        return "stats"  # TODO: prepare a proper PDF report

    def program_args(self):
        return [self.internal_script("collectStats.py"),
                self.config['imzml'],
                self.output_filename()]

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

class SimulateAndRunFullPipeline(luigi.Task):
    config = luigi.DictParameter()

    def requires(self):
        return CreateAnnotationConfigForSimulatedData(self.config)

    def _pipeline(self):
        return RunFullPipeline(simulatedDataConfig(self.config))

    def output(self):
        return self._pipeline().output()

    def run(self):
        self._pipeline().run()

# TODO
class ComputeSimilarityMetrics(luigi.WrapperTask):
    config = luigi.DictParameter()

    def requires(self):
        return [RunFullPipeline(self.config),
                CreateAnnotationConfigForSimulatedData(self.config),
                RunFullPipeline(simulatedDataConfig(self.config))]

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
