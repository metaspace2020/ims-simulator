from setuptools import setup, find_packages

from ims_simulator import __version__   

setup(name='ims_simulator',
      version=__version__,
      description='Python library for simulating a high-resolution imaging mass spectrometry data',
      url='https://github.com/metaspace2020/ims-simulator',
      author='Artem Tarasov, EMBL; Andrew Palmer, EMBL',
      packages=find_packages())
