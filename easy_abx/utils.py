#!/usr/bin/env python

''' function used by scripts in easy_abx '''

import os
import sys
import os.path
import sys
package_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not(package_path in sys.path):
    sys.path.append(package_path)
import string
import random
from collections import namedtuple


import pandas as pd
import numpy as np
import scipy.spatial.distance
import sklearn.metrics.pairwise
from pyparsing import Word, nums, Suppress, Literal, Group, delimitedList
import h5py


import ABXpy.task
import ABXpy.distances.distances  
import ABXpy.distances.distances as distances
import ABXpy.distances.metrics.cosine as cosine
import ABXpy.distances.metrics.dtw as dtw
import ABXpy.score as score
import ABXpy.misc.items as items
import ABXpy.analyze as analyze


__all__ = ['create_abx_files', 'parse_ranges']  
__all__ += ['cosine_distance', 'compute_abx']


def create_abx_files(df, options, output_name):
    """ it create item and feature files used by ABXpy from a csv files
   
        Parameters
        ----------

        df :  
            pandas dataframe with labels in the first label
        
        options : 
            a namedtuple that contains the fields
            col_items - contain the indexes of columns with item values 
                        items are the experiment name, name of the pearson linked with
                        a response, etc (alphanumeric)
            col_coords -  the position in the file where the features are extracted (numeric, optional) 
            col_labels - the label for the feature or response or experiment (alphanumeric) 
            col_features - contain the features or responses (numerical)

        output_name : 
            file name of the item (text) an feature files (h5) 

        Returns
        -------

        two files with names "output_name".item and "output_name".features


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
    features = df.iloc[:,options.col_features].values.astype(np.float64)

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
    [2, 2, 3]

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


# from https://goo.gl/REx446
def __id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

# FIXME : change this monkey patching!
# This class override ABXpy.distances.distances.Features_Accessor
# in order to avoid using the timestamps in the items/feature files
class Modified_Features_Accessor(ABXpy.distances.distances.Features_Accessor):

    def __init__(self, times, features):
        self.times = times
        self.features = features

    def get_features_from_raw(self, items):
        features = {}
        for ix, f, on, off in zip(items.index, items['file'],
                                  items['onset'], items['offset']):
            f=str(f)
            #t = np.where(np.logical_and(self.times[f] >= on,
            #                            self.times[f] <= off))[0]
            features[ix] = self.features[f]#[t, :]
        return features

ABXpy.distances.distances.Features_Accessor = Modified_Features_Accessor

# /!\ ABXpy distance needs always a third argument !
def cosine_similarity(x, y, normalized):
    return sklearn.metrics.pairwise.cosine_similarity(x, y)

def cosine_distance(x, y, normalized):
    #return sklearn.metrics.pairwise.cosine_distances(x, y)
    #return cosine.cosine_distance(x, y)
    return scipy.spatial.distance.cosine(x, y)
    
def correlation_distance(x, y, normalized):
    return scipy.spatial.distance.correlation(x, y)

def dtw_cosine_distance(x, y, normalized):
    return dtw.dtw(x, y, cosine.cosine_distance, normalized)


def run_abx(data_file, on, across, by, njobs, tmpdir=None, distance=cosine_distance):
    ''' wrap ABXpy funcions and compute the scores '''
    item_file = '{}.item'.format(data_file)
    feature_file = '{}.features'.format(data_file)
    distance_file = '{}.distance'.format(data_file)
    scorefilename = '{}.score'.format(data_file)
    taskfilename = '{}.abx'.format(data_file)
    analyzefilename = '{}.csv'.format(data_file)

    # clean up before compute ABX
    remove_files = [distance_file, scorefilename, taskfilename, analyzefilename]
    map(os.remove, filter(os.path.exists, remove_files))

    # running the evaluation
    task = ABXpy.task.Task(item_file, on, across=across, by=by, verbose=False)
    task.generate_triplets(taskfilename, tmpdir=tmpdir)
    distances.compute_distances(feature_file, '/features/', taskfilename,
                                distance_file, distance, normalized=True, n_cpu=njobs)
    score.score(taskfilename, distance_file, scorefilename)
    analyze.analyze(taskfilename, scorefilename, analyzefilename)


def compute_abx(features, labels, on, across=None, by=None, njobs=1, distance=cosine_distance):
    ''' compute_abx computes the ABX task 
   
    Parameters
    ----------

    features : 
    
    labels : 
    
    on :  
    
    across : [default=None] 
    
    by : [default=None]
    
    njobs : [default=1] 
    
    distance : [default=cosine_distance]

    Returns
    -------
    abx_scores
    
    '''

    # create intermediate files
    new_labels = np.array([labels]).T
    _, num_labels = new_labels.shape
    new_features = np.array(features, dtype=np.float64)
    _, num_features = new_features.shape
    df = pd.DataFrame(np.hstack((new_labels, new_features))) 
   
    # conf
    cmd_options = ['col_items', 'col_coords', 'col_labels', 'col_features', 'no_header']
    options = namedtuple('options', cmd_options)
    options.col_items = None  
    options.col_coords = None 
    options.col_labels = range(num_labels) 
    options.col_features = range(num_labels, num_features + num_labels)
    options.header = False
    data_file = __id_generator()
   
    # running abx
    create_abx_files(df, options, data_file)
    run_abx(data_file, on, across, by, njobs, tmpdir=None, distance=distance)
    abx_scores = pd.read_csv('{}.csv'.format(data_file), sep='\t')

    # cleaning
    item_file = '{}.item'.format(data_file)
    feature_file = '{}.features'.format(data_file)
    distance_file = '{}.distance'.format(data_file)
    scorefilename = '{}.score'.format(data_file)
    taskfilename = '{}.abx'.format(data_file)
    analyzefilename = '{}.csv'.format(data_file)
    remove_files = [distance_file, scorefilename, taskfilename, analyzefilename]
    remove_files += [item_file, feature_file]
    map(os.remove, filter(os.path.exists, remove_files))

    return abx_scores


if __name__ == "__main__":
    import doctest
    doctest.testmod()

