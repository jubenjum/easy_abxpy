[![Build Status](https://travis-ci.org/primatelang/easy_abxpy.svg?branch=master)](https://travis-ci.org/primatelang/easy_abxpy)
[![Anaconda-Server Badge](https://anaconda.org/primatelang/easy_abx/badges/installer/conda.svg)](https://conda.anaconda.org/primatelang)



The objetive of the scripts in this directory is to convert
from a csv file to item and features files required 
to run ABXpy


Installing Packages
===================

Use conda to install all the required packages, if you don't have it in your 
system follow the instructions on https://conda.io/docs/user-guide/install/index.html

With conda installed in your system, follow the commands easy_abx package:


## from the source 

	conda config --add channels primatelang
	conda create -n abx 
	source activate abx
	conda install --file requirements.txt
	python setup.py install

# from conda
        conda create -n abx
        source activate abx
        conda install -c primatelang easy_abxpy


Running examples
================

I included two examples, both in the `tests` directory. 

- tests/pca.csv: from the analysis of monkeys calls, it contains in the
                 first column the labels, and the rest of columns are 
                 the features that describe the labels 
		 (flatten 50x40 dim MFCC's to 20x1 dim PCA)

- tests/items_020_corr.csv: from an auditory experiment 

I included the script `bin/run.sh` script that shows how does it work the pipeline:
   
             csv -> csv convertion -> run abxpy

Scripts input parameters
========================

- prepare_abx.py: you will need the csv file and the name for your experiment (output file name),
  also you will need to include in the command line the columns that explain your data:

	- col_features [required] : columns numbers for the features, explanatory variable  
	- col_labels [required] : columns numbers for the label, response variable 
	- col_items [optional]: column numbers for the items, or the name of the experiment/speaker/sound
	
- run_abx.py: the input is the name of the experiment (output file name) 
  you used in prepare_abx.py
 
check all the options of each of these scripts with the -h option


Pipeline outputs
================

After running the pipeline you will have the files

- *.items: build with prepare_abx.py, uses you csv file
- *.features: HDR file created by prepare_abx.py using your csv file 
- *.abx, *.distances, *.score : outpus of ABXpy 
- *.csv: all the scores from ABXpy, ***the results***

