#!/usr/bin/env python
"""Python setup script for the wordseg package
This script is not intented to be use directly but must be configured
by cmake.
"""

import os
from setuptools import setup


setup(
    name='easy_abx',
    version='0.0.1',
    description='tools to build item/feature files and run it',
    author='Juan Benjumea, Ewan Dumbar',
    url='https://github.com/primatelang/easy_abxpy',
    license='GPL3',

    install_requires=(['six', 'joblib', 'numpy', 'pandas', 'h5py', 
        'scipy', 'pyparsing']),


    entry_points = {'console_scripts': 
         ['prepare_abx = prepare_abx:main', 
          'run_abx = run_abx:main', ],}
)


