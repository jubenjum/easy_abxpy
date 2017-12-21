#!/usr/bin/env python
"""Python setup script for the wordseg package
This script is not intented to be use directly but must be configured
by cmake.
"""

import os
from setuptools import setup

readme = open('readme.md').read() 


requirements = [  ]
dependency_links = [  ] 

setup(
    name='easy_abx',
    version='0.1.0',
    description='tools to build item/feature files and run it',
    long_description=readme + '\n\n',
    
    author='Juan Benjumea, Ewan Dumbar',
    url='https://github.com/primatelang/easy_abxpy',

    #packages=['easy_abx'],
    #package_dir={'': 'src'},

    include_package_data=True,
    install_requires=requirements,
    dependency_links=dependency_links,


    entry_points = {'console_scripts': 
         ['prepare_abx = easy_abx.prepare_abx:main', 
          'run_abx = easy_abx.run_abx:main', ],}
)


