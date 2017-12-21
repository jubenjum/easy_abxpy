#!/bin/bash

set -e

source activate abx

echo
echo "Running the calls comparision test"
echo
prepare_abx tests/pca.csv SP4050BM --col_features 2-20 --col_labels 1

run_abx SP4050BM --on "call"


echo
echo "Running articulatory experiment test"
echo
prepare_abx tests/items_020_corr.csv articulatory --header \
     --col_features 7-16 --col_labels 1,4,5,6 --col_items 3,2,1

run_abx articulatory --on "target-word" --by "place" --across "type"


source deactivate
