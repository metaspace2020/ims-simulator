import luigi

import pandas as pd

from os.path import splitext, basename, expanduser
import subprocess
import tempfile
import shutil
import json
import sys
import os

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

def get_id(obj):
    try:
        return str(abs(hash(obj)))[:8]
    except:
        # horrible hacks to get a hash for luigi.DictParameter
        def unfreeze(x):
            if type(x) == luigi.parameter.FrozenOrderedDict:
                return {k: unfreeze(x[k]) for k in x}
            else:
                return x

        if type(obj) == luigi.parameter.FrozenOrderedDict:
            obj = unfreeze(obj)

        import json
        return str(abs(hash(json.dumps(obj))))[:8]

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

    def run(self):
        args = [str(arg) for arg in self.program_args()]

        if self.output_is_stdout():
            f = tempfile.NamedTemporaryFile(delete=False)
            subprocess.check_call(args, stdout=f)
            f.close()
            shutil.move(f.name, self.output_filename())
        else:
            subprocess.check_call(args)

class CreateIsotopeDB(CmdlineTask):
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()

    def output_filename(self):
        db_name = get_prefix(self.molecular_db['sum_formulas_fn'])
        fn = "{}_{}_{}_{}.db".format(db_name,
                                     self.instrument['type'],
                                     self.instrument['res200'],
                                     get_id(self.molecular_db))
        if self.molecular_db['is_decoy']:
            fn += ".decoy"
        return fn

    def program_args(self):
        return ['ims', 'isocalc',
                '--instrument', self.instrument['type'],
                '--resolution', self.instrument['res200'],
                '--adducts', ",".join(self.molecular_db['adducts']),
                self.molecular_db['sum_formulas_fn'],
                self.output_filename()]

class ConvertImzMLToImzb(CmdlineTask):
    imzml_fn = luigi.Parameter()

    def output_filename(self):
        return splitext(self.imzml_fn)[0] + ".imzb"

    def program_args(self):
        return ['ims', 'convert',
                self.imzml_fn, self.output_filename()]

class RunAnnotation(CmdlineTask):
    imzml_fn = luigi.Parameter()
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()

    def requires(self):
        return {
            'imzb': ConvertImzMLToImzb(self.imzml_fn),
            'isotope_db': CreateIsotopeDB(self.instrument, self.molecular_db),
        }

    def output_filename(self):
        db_id = get_id(self.molecular_db) + "."
        db_id += 'decoy' if self.molecular_db['is_decoy'] else 'target'
        return get_prefix(self.imzml_fn) + ".results." + db_id

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

    def requires(self):
        return RunAnnotation(self.imzml_fn, self.instrument, self.molecular_db)

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
            'target_results': GetAnnotationsForAdduct(self.imzml_fn, self.instrument, self.molecular_db, self.adduct),
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
        return self.config['database']

    def depends_on(self):
        return [k for k in self.config.keys() if k != 'imzml']

    def file_extension(self):
        raise NotImplementedError

    def output_filename(self):
        fn = get_prefix(self.imzml_fn)
        for key in self.depends_on():
            fn += "_" + get_id(self.config[key])
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

    def file_extension(self):
        return "layers"

    def program_args(self):
        return [self.internal_script("assignMolecules.py"),
                self.input().fn, self.output_filename(),
                "--instrument", self.instrument['type'],
                "--res200", self.instrument['res200'],
                "--db", self.molecular_db['sum_formulas_fn']]

class SimulateDataset(SimulationTask):
    def requires(self):
        return AssignMolecules(self.config)

    def file_extension(self):
        return "simulated.imzML"

    def program_args(self):
        return [self.internal_script("simulate.py"),
                self.input().fn, self.output_filename(),
                "--instrument", self.instrument['type'],
                "--res200", self.instrument['res200']]

class RunFullPipeline(luigi.Task):
    config = luigi.DictParameter()

    def requires(self):
        imzml = self.config['imzml']
        instrument = self.config['instrument']
        db = self.config['database']

        return {adduct: ComputeFDR(imzml, instrument, db, adduct)
                for adduct in db['adducts']}

    def output(self):
        return {adduct: luigi.LocalTarget(self.input()[adduct].fn)
                for adduct in self.input()}

    def run(self):
        pass

class SimulateAndRunFullPipeline(luigi.WrapperTask):
    config = luigi.DictParameter()

    def requires(self):
        return SimulateDataset(self.config)

    def run(self):
        simulation_config = luigi.parameter.FrozenOrderedDict(
            imzml=self.input().fn,
            database=self.config['database'],
            instrument=self.config['instrument']
        )
        print simulation_config
        yield RunFullPipeline(simulation_config)

# TODO
class ComputeSimilarityMetrics(luigi.WrapperTask):
    config = luigi.DictParameter()

    def requires(self):
        return [RunFullPipeline(config),
                SimulateAndRunFullPipeline(config)]

def readConfig(filename):
    with open(filename) as conf:
        config = json.load(conf, object_pairs_hook=luigi.parameter.FrozenOrderedDict)

    cfg = config.get_wrapped()
    cfg['imzml'] = expanduser(config['imzml'])
    cfg['database'].get_wrapped()['sum_formulas_fn'] = expanduser(config['database']['sum_formulas'])
    cfg['database'].get_wrapped()['adducts'] = tuple(cfg['database']['adducts'])
    cfg['database'].get_wrapped()['is_decoy'] = False
    return config

if __name__ == '__main__':
    config = readConfig(sys.argv[1])
    luigi.build([ComputeSimilarityMetrics(config)])
