#!/usr/bin/env python

"""
compute_abx_on: compute ABX scoref or the ON task using a short version 
of ABX algorithm.
"""

import sys
import pandas as pd
import numpy as np
from collections import namedtuple


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

    parser.add_argument('--header', action='store_true', 
            help='first line in the csv file is a header')

    args = parser.parse_args()

    # in and out files
    input_csv = args.csv_file

    # it's easier to count columns from one, I transform that to pythons indexes (0 based)
    ranges = lambda x: list(np.array(parse_ranges(x))-1)
    col_on = ranges(args.col_on) if args.col_on else None
    col_features = ranges(args.col_features[0]) if args.col_features else None
    header = args.header

    is_matrix = True
    if header:
        try:
            df = pd.read_csv(input_csv)
        except pd.errors.ParserError:
            is_matrix = False

    else:
        try:
            df = pd.read_csv(input_csv, header=None)
        except pd.errors.ParserError:
            is_matrix = False

    if not is_matrix:
        print('ABX score is computed from fixed length features' +
                ': {} has mixed lengths'.format(input_csv))
        sys.exit()


    features = df.iloc[:, col_features].values
    labels = df.iloc[:, col_on].values
    abx_by_on(features, labels)

if __name__ == '__main__':
    main()
