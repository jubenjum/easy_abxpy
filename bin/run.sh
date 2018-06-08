#!/bin/bash

set -e

#source activate abx

echo
echo "Running the calls comparision test"
echo
prepare_abx "test/pca.csv" SP4050BM --col_features 2-20 --col_labels 1

run_abx SP4050BM --on "call" --distance "distances/euclidean_distance.py" 


echo
echo "Running articulatory experiment test"
echo
prepare_abx "test/items_020_new.csv" articulatory --header \
     --col_features 8-16 --col_labels 1,4-7 --col_items 3,2,1

run_abx articulatory --distance "" --on "target-word" --by "place" "position-wd" --across "type"


source deactivate
