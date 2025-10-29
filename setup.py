from setuptools import setup, find_packages

setup(
    name='dd4hep_hit_analysis_framework',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'uproot',
        'numpy',
        'hist',
        'awkward',
        'matplotlib',
        'mplhep'
    ],
    author='Dimitris Ntounis',
    author_email='dntounis@stanford.edu',
    description='Framework for hit-level analysis in dd4hep',
    url='https://github.com/dntounis/dd4hep_hit_analysis_framework',
)
