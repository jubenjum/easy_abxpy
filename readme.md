
The objetive of the scripts in this directory is to convert
from a csv file to the item and features files required 
to run ABXpy


Install Packages
================

Use conda to install all the packages, if you don't have it in your 
system follow the instructions from https://conda.io/docs/user-guide/install/index.html

With conda installed in your system follow the commands 
to install all packages used used by this package: 

	conda create -n abx 
	source activate abx
	conda install --file requirements.txt
	pip install git+https://github.com/bootphon/h5features@82aeee8c3072f89276d99d7593a2f1c7a2f3719c
	pip install git+https://github.com/bootphon/ABXpy@7ee2783ac3fea906c436391ccba74422157b3c28


Run examples
============

I included two examples, both in the `tests` directory. 

- tests/pca.csv: from the analysis of monkeys calls, it contains in the
                 first column the labels, and the rest of columns are 
                 the features that describe the labels

- tests/items_020_corr.csv: from ....

I included the script `run.sh` script that shows the pipeline:
   
             csv -> csv convertion -> run abxpy

Scripts input parameters
========================

- prepare_abx.py: you will need the csv file and a name for your experiment,
  you need to include in the command line the columns that that explain your data:

	- col_features [required] : features, explanatory variable  
	- col_labels [required] : label, response variable 
	- col_items [optional]: the name of the experiment

- run_abx.py: it will need the name of the experiment you used in prepare_abx.py
 

check all the options of each of these scripts with the -h option, and check the 


Pipeline outputs
================

After running the pipeline you will have the files

- *.items: build with prepare_abx.py, uses you csv file
- *.features: HDR file created by prepare_abx.py using your csv file 
- *.abx, *.distances, *.score : outpus of ABXpy 
- *.csv: all the scores from ABXpy, ***the results***

