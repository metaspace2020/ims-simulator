from setuptools import setup, find_packages

from ims_simulator import __version__   

setup(name='ims_simulator',
      version=__version__,
      description='Python library for bootstrapping a high resolution imaging dataset',
      url='https://github.com/spatialmetabolomics/ims_simulator',
      author='Artem Tarasov, EMBL; Andrew Palmer, EMBL',
      packages=find_packages())
