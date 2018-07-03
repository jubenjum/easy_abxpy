# ABX discrimination task
[![Build Status](https://travis-ci.org/primatelang/easy_abxpy.svg?branch=master)](https://travis-ci.org/primatelang/easy_abxpy)
[![Anaconda-Server Badge](https://anaconda.org/primatelang/easy_abx/badges/installer/conda.svg)](https://conda.anaconda.org/primatelang)

Scripts that computes ABX discrimination task by wrapping
[ABXpy](https://github.com/bootphon/ABXpy) for complex cases
and a light version of ABXpy code to compute scores for simple cases, 
all inputs are csv files.


# Installation

To use the package you will need a python 2.7 distribution in you system.
The easiest way to have all running is to use
[anaconda](https://www.anaconda.com/download/), as it manages all package
requirements.

With conda installed in your system, you can install *easy_abx* from the
source or using conda packages, follow one of the following methods to 
install easy_abx package:


## Method 1: from source 

You will need to get the source:

    $ git clone https://github.com/primatelang/easy_abxpy
    $ cd easy_abxpy

Optionally you can create and load a conda virtual environment, that 
will keep isolated this package and its requirements form other packages:


    $ conda create -n abx 
	$ source activate abx

Next, install the required packages:

	$ conda install --file requirements.txt

Finally run the installation script:

    $ python setup.py install

## Method 2: conda

Conda will install all requirements for you, as before you can also create 
and load a conda virtual environment to keep isolated the package.  
To install the package do:

    $ conda install -c primatelang easy_abx

The supported architectures for this package are modern Linux and OSX distributions.


# Examples

Two examples are included in the *tests* directory. 

- *tests/pca.csv*: an analysis from done for monkeys calls. The data are Mel filter-bank
spectral features projected to a 20D using Principal Component Analysis (PCA). The 
csv-file contains 21 columns, the fist column has the call **labels**, and the rest 
of columns are the compressed **features** 
  
- *tests/items_020_corr.csv*: from an auditory experiment 


The script **bin/run.sh** runs the pipeline in example datasets and shows does the 
package and scripts work. But essentially you can compute ABX in two ways with this 
package:

1. By using ABXpy wrappers: using `prepare_abx` and `run_abx`, it will generate 
all needed [ABXpy files](http://abxpy.readthedocs.io/en/latest/ABXpy.html#the-pipeline)
and run that pipeline, for the *pca.csv* example:

    $ prepare_abx "test/pca.csv" EXPNAME --col_features 2-21 --col_labels 1
    
    $ run_abx EXPNAME --on "call"

2. computing only [ON task](http://abxpy.readthedocs.io/en/latest/ABXpy.html#ABXpy.task.Task)
with a lightweight ABX implementation, running the same example:

    $ compute_abx "test/pca.csv" --col_on 1 --col_features 2-21 > EXPNAME_LABX.csv

For more information about how to use the scripts, see their help messages.


# Scripts input parameters

- *prepare_abx.py*: This script needs a csv-file with the input data, and the name 
of the experiment that you want to give to the output file, other parameters used in
the command line and that explain your data are:

	- *col_features*: numeric and required. 
           columns numbers for the features, explanatory variable  

	- *col_labels*: numeric and required  
           columns numbers for the label, response variable 
	
    - *col_items*: optional numerical parameter
           column numbers for the items, or the name of the experiment/speaker/sound
	
- *run_abx.py*: the input is the name of the experiment (output file name) 
you used in *prepare_abx.py*
 
check all the options of each of these scripts with the -h option


# Pipeline outputs

After running the pipeline you will have the files

- *.items: build with prepare_abx.py, uses you csv file
- *.features: HDR file created by prepare_abx.py using your csv file 
- *.abx, *.distances, *.score : outpus of ABXpy 
- *.csv: all the scores from ABXpy, ***the results***

