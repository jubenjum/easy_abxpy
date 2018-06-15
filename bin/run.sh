#!/bin/bash

set -e

#source activate abx

# examples of running ABX analysis.
#
# The first example computes ABX scores for acoustic Blue monkeys cries,
# the data consist in 40 filter banks 50frames features by call, 
# these multidimensional features where then shrink using PCA. These 
# two calls are in the corpus: "p" and "h", and are keep the file pca.csv 
# in the column 1, the rest of columns are the shrink features.
#
# The distance used by ABXpy is defined in the 'distances/euclidean_distance.py' 
# python file.
#

echo
echo "Running the calls comparision test"
echo
prepare_abx "test/pca.csv" SP4050BM_ABXpy --col_features 2-20 --col_labels 1

run_abx SP4050BM_ABXpy --on "call" --distance "distances/euclidean_distance.py" 

# compute abx with for on task with using only python
compute_abx "test/pca.csv" --col_on 1 --col_features 2-20 \
    --distance "distances/euclidean_distance.py" > SP4050BM_easyabx.csv

#
# this second example uses data from an arituculatory experiment, and it uses 
# a combination of ABX scores for by and across tasks
#

echo
echo "Running articulatory experiment test"
echo
prepare_abx "test/items_020_new.csv" articulatory --header \
     --col_features 8-16 --col_labels 1,4-7 --col_items 3,2,1

run_abx articulatory --on "target-word" --by "place" "position-wd" --across "type"

#source deactivate
