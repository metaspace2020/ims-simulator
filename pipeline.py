import luigi

import pandas as pd

from os.path import splitext, basename, expanduser
import subprocess
import json

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
    return str(abs(hash(obj)))[:8]

class CreateIsotopeDB(luigi.Task):
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()

    def _output_filename(self):
        db_name = splitext(basename(self.molecular_db['sum_formulas_fn']))[0]
        fn = "{}_{}_{}_{}.db".format(db_name,
                                     self.instrument['type'],
                                     self.instrument['res200'],
                                     get_id(self.molecular_db))
        if self.molecular_db['is_decoy']:
            fn += ".decoy"
        return fn

    def output(self):
        return luigi.LocalTarget(self._output_filename())

    def run(self):
        cmd = ['ims', 'isocalc',
               '--instrument', self.instrument['type'],
               '--resolution', str(self.instrument['res200']),
               '--adducts', ",".join(self.molecular_db['adducts']),
               self.molecular_db['sum_formulas_fn'],
               self._output_filename()]
        subprocess.check_call(cmd)

class ConvertImzMLToImzb(luigi.Task):
    imzml_fn = luigi.Parameter()

    def _output_filename(self):
        return splitext(self.imzml_fn)[0] + ".imzb"

    def output(self):
        return luigi.LocalTarget(self._output_filename())

    def run(self):
        cmd = ['ims', 'convert',
               self.imzml_fn, self._output_filename()]
        subprocess.check_call(cmd)

class RunAnnotation(luigi.Task):
    imzml_fn = luigi.Parameter()
    instrument = luigi.DictParameter()
    molecular_db = luigi.DictParameter()

    def requires(self):
        return {
            'imzb': ConvertImzMLToImzb(self.imzml_fn),
            'isotope_db': CreateIsotopeDB(self.instrument, self.molecular_db),
        }

    def _output_filename(self):
        db_id = get_id(self.molecular_db) + "."
        db_id += 'decoy' if self.molecular_db['is_decoy'] else 'target'
        return splitext(basename(self.imzml_fn))[0] + ".results." + db_id

    def output(self):
        return luigi.LocalTarget(self._output_filename())

    def run(self):
        cmd = ['ims', 'detect',
               self.input()['isotope_db'].fn,
               self.input()['imzb'].fn]
        with open(self._output_filename(), "w+") as f:
            subprocess.check_call(cmd, stdout=f)

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

class ComputeFDR(luigi.Task):
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

    def _output_filename(self):
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

        return tuple(list(result))

    def _decoy_molecular_db(self):
        return luigi.parameter.FrozenOrderedDict([
            ('sum_formulas_fn', self.molecular_db['sum_formulas_fn']),
            ('adducts', self._decoy_adducts()),
            ('is_decoy', True)
        ])

    def output(self):
        return luigi.LocalTarget(self._output_filename())

    def run(self):
        cmd = ['ims', 'fdr',
               self.input()['target_results'].fn,
               self.input()['decoy_results'].fn]
        if self.groundtruth_fn:
            cmd += ['--groundtruth', self.groundtruth_fn]
        with open(self._output_filename(), "w+") as f:
            subprocess.check_call(cmd, stdout=f, stderr=subprocess.STDOUT)

class RunFullPipeline(luigi.WrapperTask):
    config = luigi.DictParameter()

    def requires(self):
        imzml = self.config['imzml']
        instrument = self.config['instrument']
        db = self.config['database']

        return [ComputeFDR(imzml, instrument, db, adduct)
                for adduct in db['adducts']]

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
    import sys
    config = readConfig(sys.argv[1])
    luigi.build([RunFullPipeline(config)], scheduler_port=8083)
