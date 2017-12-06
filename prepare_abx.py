#!/usr/bin/env python

"""
prepare_abx: prepare the abx files to compute ABX score.
"""


import sys
from collections import namedtuple

import pandas as pd
import numpy as np
from pyparsing import Word, nums, Suppress, Literal, Group, delimitedList
import h5py


def create_abx_files(df, options, output_name):
    """ it create item and feature files from a csv
       
        options contains the fields:
        col_items: contain the indexes of columns with item values 
                   items are the experiment name, name of the pearson linked with
                   a response, etc (alphanumeric)
        col_coords: the position in the file where the features are extracted (numeric, optional) 
        col_labels: the label for the feature or response or experiment (alphanumeric) 
        col_features : contain the features or responses (numerical)
    """
  
    # input files for abx ...
    item_file = '{}.item'.format(output_name)
    features_file = '{}.features'.format(output_name)

    num_exp, num_vars = df.shape

    # extracting the file names (items) from csv
    joined_items = []
    if options.col_items is None: # mock the item/filename if not given
        joined_items = ['item_{:04d}'.format(x) for x in range(1, num_exp+1)]
    else:
        items = df.iloc[:,options.col_items].values
        if len(items) > 1:
            for item in items:
                join_item = '_'.join(['{}'.format(x) for x in item])
                joined_items.append(join_item)
                
        else:
            joined_items = ['{}'.format(x) for x in items] 

    # extracting the labels, in abx are the interval times linked
    # to the sound file, however in experiments without these it can 
    # be fill with random values 
    times = []
    if options.col_coords is None:
        coords = [np.random.rand(2,1) for _ in range(num_exp)]
    else:
        coords = df.iloc[:,options.col_coords].values
    
    # extracting the labels, in this case I use the machine learning definition 
    # of labeling (tags)
    labels = df.iloc[:,options.col_labels].values  

    # extracting the features
    features = df.iloc[:,options.col_features].values 

    # indexes links the features with the items (items <- indexes -> features)
    # index(es) are linked one to one with item in items, and the number in
    # the value of index is the first position in the features for the given item 
    index = np.arange(start=0, stop=num_exp) # if features are flat, one row for one item

    ### generating the features file
    f = h5py.File(features_file, "w")
    group = f.create_group("features")
    group.attrs['version'] = '1.1'
    group.attrs['format'] = 'dense'
    features_ = group.create_dataset("/features/features/", data=features)
    index_ = group.create_dataset("/features/index/", data=index)
    items_ = group.create_dataset("/features/items/", data=joined_items) 
    labels_ = group.create_dataset("/features/labels/", data=coords) 
    f.close()

    ### generate the item file
    with open(item_file, 'w') as ifile:
        if options.header:
            header = list(df.columns[[options.col_labels]]) 
            left = "#file onset offset #" # ? always the same 
            right = " ".join(["{}".format(x) for x in header])
            item_header = left + right + "\n" 

        else:
            num_labels = len(labels[0])
            item_header = "#file onset offset #" + "call"*num_labels + "\n"
        
        ifile.write(item_header)
        
        for join_item, coord, label in zip(joined_items, coords, labels):
            joined_labels = ' '.join(['{}'.format(x) for x in label])
            joined_coords = ' '.join(['{}'.format(x[0]) for x in coord])
            t = '{} {} {}\n'.format(join_item, joined_coords, joined_labels)
            ifile.write(t)


def parse_ranges(text):
    """parse simple numeric range expression
    
    It allows to trasform ranges to python list. The input range is a string, 
    and it will return the a list filled with integers in a range, for example

    >>> parse_ranges('1')
    [1]

    >>> parse_ranges('2,2,3')
    [1, 2, 3]

    >>> parse_ranges('3,0-4')
    [3, 0, 1, 2, 3, 4]
    
    """
    
    if not text:
        return []

    # single integer value
    number = Word(nums).setParseAction(lambda t: int(t[0]))
    
    # range values
    ran_sep = Suppress(Literal('-'))
    num_range = Group(number("start") + ran_sep + number("end"))
    
    # grammar + decode
    expr = num_range | number
    exprGramm = delimitedList(expr)
    decoded_text = exprGramm.parseString(text)

    selected_ranges = []
    for n in decoded_text:
        if isinstance(n, int): # single value
            selected_ranges.append(n)
        else: # range of values
            if n.end < n.start:
                print("Error in ranges, they should be start<end")
                raise 
            if n.start == n.end:
                selected_ranges.append(int(n.start))
            else:
                selected_ranges += range(n.start, n.end+1)

    return selected_ranges


def one2zero(a):
    """removes one to  """
    return list(np.array() - 1 )


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
    
    ##print args.col_items, parse_ranges(args.col_items), options.col_items
    ##print args.col_coords, parse_ranges(args.col_coords), options.col_coords
    ##print args.col_labels, parse_ranges(args.col_labels), options.col_labels
    ##print args.col_features[0], parse_ranges(args.col_features[0]), options.col_features

    if options.header:
        df = pd.read_csv(input_csv)
    else:
        df = pd.read_csv(input_csv, header=None)

    create_abx_files(df, options, output_name)


if __name__ == '__main__':
    main()
