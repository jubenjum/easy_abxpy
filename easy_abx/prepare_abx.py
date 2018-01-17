#!/usr/bin/env python

"""
prepare_abx: prepare the abx files to compute ABX score.
"""

import sys
import pandas as pd
import numpy as np
from collections import namedtuple


from utils import create_abx_files
from utils import parse_ranges


def main():
    import argparse

    parser = argparse.ArgumentParser( prog=sys.argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='prepare item/features abx files to compute ABX score')

    parser.add_argument('csv_file', help='input csv file')

    parser.add_argument('output_name', help='experiment name, output file name of abx files')

    parser.add_argument('--col_items', help='columns with the file name or items, columns begins in 1')

    parser.add_argument('--col_coords', help='columns with the coords, columns begins in 1')

    parser.add_argument('--col_labels', required=True,
            help='columns with the labels, columns begins in 1')

    parser.add_argument('--col_features', nargs='*', required=True,
            help='columns with the features name, columns begins in 1')

    parser.add_argument('--header', action='store_true', help='first line in the csv file is a header')

    args = parser.parse_args()

    # in and out files
    input_csv = args.csv_file
    output_name = args.output_name if args.output_name else 'example'

    # options to pass to  create_abx_files function
    cmd_options = ['col_items', 'col_coords', 'col_labels', 'col_features', 'no_header']
    options = namedtuple('options', cmd_options)

    # it's easier to count columns from one, I transform that to pythons indexes (0 based)
    ranges = lambda x: list(np.array(parse_ranges(x))-1)
    options.col_items = ranges(args.col_items) if args.col_items else None
    options.col_coords = ranges(args.col_coords) if args.col_coords else None
    options.col_labels = ranges(args.col_labels) if args.col_labels else None
    options.col_features = ranges(args.col_features[0]) if args.col_features else None
    options.header = args.header

    is_matrix = True
    if options.header:
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

    create_abx_files(df, options, output_name)


if __name__ == '__main__':
    main()
