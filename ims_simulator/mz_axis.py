class Instrument(object):
    def __init__(self, args):
        self.instrument = args.instrument
        self.res200 = args.res200

    def resolutionAt(self, mz):
        if self.instrument == 'orbitrap':
            return self.res200 * (200.0 / mz) ** 0.5
        elif self.instrument == 'fticr':
            return self.res200 * (200.0 / mz)

def generate_mz_axis(mz_min, mz_max, instrument, step_size=5):
    """
    returns array of non-overlapping tuples (mz, ppm) that cover [mz_min, mz_max]
    """
    mz_axis = []
    mz = mz_min
    while mz < mz_max:
        fwhm = mz / instrument.resolutionAt(mz)
        step = step_size * fwhm
        ppm = 1e6 * step / (2.0 * mz + step)
        mz_axis.append((mz + step/2, ppm))
        mz += step
    return mz_axis
