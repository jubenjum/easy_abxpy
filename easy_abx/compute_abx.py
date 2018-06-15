#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compute_abx_on: compute ABX scoref or the ON task using a short version 
of ABX algorithm.
"""

import sys
import imp

import pandas as pd
import numpy as np


try:
    # python 3.6
    from pathlib import Path
except ImportError:
    # python 2.6
    from pathlib2 import Path

from scipy.spatial.distance import euclidean

from utils import abx_by_on
from utils import parse_ranges


def main():
    import argparse

    parser = argparse.ArgumentParser( prog=sys.argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='compute ABX score using a light version of abx algorithm.')

    parser.add_argument('csv_file', help='input csv file')

    parser.add_argument('--col_on', required=True,
            help='column where is keep the on labels, counting from 1')

    parser.add_argument('--col_features', nargs='*', required=True,
            help='columns with the features name, counting from 1')

    parser.add_argument('--distance', required=False,
                        help=('uses the distnace function defined on the' 
                              ' distance file given on this argument.'
                              ' default distance function will be '))

    parser.add_argument('--header', action='store_true', 
            help='first line in the csv file is a header')

    args = parser.parse_args()


    # it's easier to count columns from one, I transform that to pythons indexes (0 based)
    ranges = lambda x: list(np.array(parse_ranges(x))-1)
    col_on = ranges(args.col_on) if args.col_on else None
    col_features = ranges(args.col_features[0]) if args.col_features else None
    
    if args.distance:
        distance_file = Path(args.distance)
        if distance_file.is_file():
            mod = imp.load_source("distance", args.distance) 
            distance = mod.distance
        else:
            print("ERROR: '{}' doesn't exist".format(args.distance))
            sys.exit()

        if not "distance" in dir():
            print("No function 'distance' in file {}".format(args.distance))
            sys.exit()
    else:
        distance = euclidean

    is_matrix = True
    if args.header:
        try:
            df = pd.read_csv(args.csv_file)
        except pd.errors.ParserError:
            is_matrix = False
    else:
        try:
            df = pd.read_csv(args.csv_file, header=None)
        except pd.errors.ParserError:
            is_matrix = False

    if not is_matrix:
        print('ABX score is computed from fixed length features' +
                ': {} has mixed lengths'.format(args.csv_file))
        sys.exit()

    features = df.iloc[:, col_features].values
    labels = df.iloc[:, col_on].values
    abx_by_on(features, labels, distance=distance)

if __name__ == '__main__':
    main()
