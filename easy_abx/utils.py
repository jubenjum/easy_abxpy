#!/usr/bin/env python

''' function used by scripts in easy_abx '''

import os
import sys
import string
import random
import hashlib
import os.path
import inspect
from collections import namedtuple, namedtuple, defaultdict
from itertools import product, permutations, combinations, count

from joblib import Memory
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
__all__ += ['get_cache_dir', 'build_cache']
__all__ += ['abx_by_on']


# Module configuration
np.random.seed(1) # initialize seed for reproductibility

package_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not(package_path in sys.path):
    sys.path.append(package_path)


# creating a global cache to store intermediate resutls
def get_cache_dir():
    """
    `get_cache_dir` function gives the directory where the cache is stored, 
    the name of the cache directory is ".cache/" and it's keep in the directory 
    where the script was called. 
    
    If ".cache" doesn't exist, this function will create it.
    """
    cdir = os.curdir+'/.cache'
    if not os.path.exists(cdir):
        os.makedirs(cdir)
    return cdir


def build_cache():
    """ `build_cache` is a shortcut for joblib.Memory function """
    return Memory(cachedir=get_cache_dir(), verbose=0)


# intializing cache
memory = build_cache()


@memory.cache
def memerized_create_abx_files(df, set_header, col_items, col_coords,
                               col_labels, col_features, output_name):
    """ it create item and feature files used by ABXpy from a csv files

        Parameters
        ----------
        df : pandas dataframe with labels in the first row and features
             from the second column ...
        set_header : Bool for writing the header in the ABXpy items file
        col_items : contain the indexes of columns with item values
                    items are the experiment name, name of the pearson linked
                    with a response, etc (alphanumeric)
        col_coords : the position in the file where the features are
                     extracted (numeric, optional)
        col_labels : the label for the feature or response or experiment
                     (alphanumeric)
        col_features : contain the features or responses (numerical)
        output_name : file name of the item (text) an feature files (h5)

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
    if col_items is None:  # mock the item/filename if not given
        joined_items = ['item_{:04d}'.format(x) for x in range(1, num_exp+1)]
    else:
        items = df.iloc[:, col_items].values
        if len(items) > 1:
            for item in items:
                join_item = '_'.join(['{}'.format(x) for x in item])
                joined_items.append(join_item)

        else:
            joined_items = ['{}'.format(x) for x in items]

    # extracting the labels, in abx are the interval times linked
    # to the sound file, however in experiments without these it can
    # be fill with random values
    if col_coords is None:
        coords = [np.random.rand(2, 1) for _ in range(num_exp)]
    else:
        coords = df.iloc[:, col_coords].values

    # extracting the labels, in this case I use the machine learning definition
    # of labeling (tags)
    labels = df.iloc[:, col_labels].values

    # extracting the features
    features = df.iloc[:, col_features].values.astype(np.float64)

    # indexes links the features with the items (items <- indexes -> features)
    # index(es) are linked one to one with item in items, and the number in
    # the value of index is the first position in the features for the given item

    # if features are flat, one row for one item in the index
    index = np.arange(start=0, stop=num_exp)

    # generating the features file
    f = h5py.File(features_file, "w")
    group = f.create_group("features")
    group.attrs['version'] = '1.1'
    group.attrs['format'] = 'dense'

    # minimum set of variables used by ABXpy
    group.create_dataset("/features/features/", data=features)
    group.create_dataset("/features/index/", data=index)
    group.create_dataset("/features/items/", data=joined_items)
    group.create_dataset("/features/labels/", data=coords)
    f.close()

    # generate the item file
    with open(item_file, 'w') as ifile:
        if set_header:
            header = list(df.columns[[col_labels]])
            left = "#file onset offset #"  # ? always the same
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

    item_data = open(item_file, 'r').read()
    features_data = open(features_file, 'rb').read()

    return (item_data, features_data)


def create_abx_files(df, options, output_name):
    """ it create item and feature files used by ABXpy from a csv files

        Parameters
        ----------

        df :
            pandas dataframe with labels in the first label

        options :
            a namedtuple that contains the fields
            col_items - contain the indexes of columns with item values
                        items are the experiment name, name of the pearson
                        linked with a response, etc (alphanumeric)
            col_coords - the position in the file where the features are
                         extracted (numeric, optional)
            col_labels - the label for the feature or response or
                         experiment (alphanumeric)
            col_features - contain the features or responses (numerical)

        output_name :
            file name of the item (text) an feature files (h5)

        Returns
        -------

        two files with names "output_name".item and "output_name".features


    """

    set_header = options.header
    col_items = options.col_items
    col_coords = options.col_coords
    col_labels = options.col_labels
    col_features = options.col_features

    # mm = memerized_create_abx_files.call_and_shelve
    mm = memerized_create_abx_files.call_and_shelve
    abx_mem_files = mm(df, set_header, col_items, col_coords, col_labels,
                       col_features, output_name)

    # save the item and features files that are keep in the cache
    item_file = '{}.item'.format(output_name)
    features_file = '{}.features'.format(output_name)
    item_data, features_data = abx_mem_files.get()
    with open(item_file, 'w') as ifile, open(features_file, 'wb') as ffile:
        ifile.write(item_data)
        ffile.write(features_data)


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
        if isinstance(n, int):  # single value
            selected_ranges.append(n)
        else:  # range of values
            if n.end < n.start:
                print("Error in ranges, they should be start<end")
                raise
            if n.start == n.end:
                selected_ranges.append(int(n.start))
            else:
                selected_ranges += range(n.start, n.end+1)

    return selected_ranges


def __id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Random string generation with upper case letters and digits in Python
    from https://goo.gl/REx446 
    """
    return ''.join(random.choice(chars) for _ in range(size))


# FIXME : change this monkey patching!
# This class override ABXpy.distances.distances.Features_Accessor
# in order to avoid using the timestamps in the items/feature files
class Modified_Features_Accessor(ABXpy.distances.distances.Features_Accessor):

    def __init__(self, times, features):
        self.times = times
        self.features = features

    def get_features_from_raw(self, items):
        """get_features_from_raw data override
        override ABXpy.distances.distances.Features_Accessor.get_features_from_raw
        removing the use of the times in the items/features file
        """
        features = {}
        for ix, f, on, off in zip(items.index, items['file'],
                                  items['onset'], items['offset']):
            f = str(f)
            # t = np.where(np.logical_and(self.times[f] >= on,
            #                            self.times[f] <= off))[0]
            features[ix] = self.features[f]  # [t, :]
        return features


ABXpy.distances.distances.Features_Accessor = Modified_Features_Accessor


# /!\ ABXpy distance needs always a third argument !
def cosine_similarity(x, y, normalized):
    return sklearn.metrics.pairwise.cosine_similarity(x, y)


def cosine_distance(x, y, normalized):
    # return sklearn.metrics.pairwise.cosine_distances(x, y)
    # return cosine.cosine_distance(x, y)
    return scipy.spatial.distance.cosine(x, y)


def correlation_distance(x, y, normalized):
    return scipy.spatial.distance.correlation(x, y)


def dtw_cosine_distance(x, y, normalized):
    return dtw.dtw(x, y, cosine.cosine_distance, normalized)


# FIXME @memory.cache crash when passing distance from the command line
#@memory.cache
def memorizable_abx(data_file, on, across, by, njobs=1, tmpdir=None,
                    distance=cosine_distance, item_features_hash='0'):
    '''
    Wrap ABXpy functions and compute the scores, this is the memorized (cached) 


    Parameters
    ----------

    data_file : [string]
        name of the item and features files without the extensionn
        these files should exist, and the name of the 
        files are : `{data_file}.features` and `{data_file}.items`

    on : [string]
        the name of the column containing the ON data. It must be 
        exist in featrues and items files

    across : [list]
        a list that contain the ACROSS column names, the elements of the 
        list are strings and need to be present on the  `{data_file}.features` 
        and `{data_file}.items`, it can be empty if this task is not 
        computed

    by : [list]
        a list that contains the BY column names as strings, the names
        must be present in the features and items files or be empty if
        BY task is not been computed

    njobs : [int, default: 1]
        number of jobs passed to joblib to compute ABX 

    tmpdir : [string, default:None] 

    distance : [function, default: cosine_distance]

    item_features_hash  [string, default:'0']
    
    Returns
    -------
    String with a the ABX results, the result is a csv string with 
    multiple lines (separated by \\n) and fields separated by tabs (\\t)
    '''
    item_file = '{}.item'.format(data_file)
    feature_file = '{}.features'.format(data_file)
    if not os.path.isfile(item_file) or not os.path.isfile(feature_file):
        raise ValueError('item_file or feature_file doesnt exist')

    # name of output files
    distance_file = '{}.distance'.format(data_file)
    score_file = '{}.score'.format(data_file)
    task_file = '{}.abx'.format(data_file)
    analyze_file = '{}.csv'.format(data_file)

    # clean up before compute ABX
    remove_files = [distance_file, score_file, task_file, analyze_file]
    map(os.remove, filter(os.path.exists, remove_files))

    # running the evaluation
    task = ABXpy.task.Task(item_file, on, across=across, by=by, verbose=False)
    task.generate_triplets(task_file, tmpdir=tmpdir)

    # distance 
    if not 'normalized' in inspect.getargspec(distance).args:
        def _distance(x, y, normalized):
            return distance(x, y)
    else:
        _distance = distance
    distances.compute_distances(feature_file, '/features/', task_file,
                                distance_file, _distance, normalized=True,
                                n_cpu=njobs)

    score.score(task_file, distance_file, score_file)
    analyze.analyze(task_file, score_file, analyze_file)

    # I will keep only the ABX scores
    remove_files = [distance_file, score_file, task_file]
    map(os.remove, filter(os.path.exists, remove_files))

    analyze_data = open(analyze_file, 'r').read()
    return analyze_data


# https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
def md5(fname):
    """`md5` function compute md5sum of a file, the objective is to 
    make include the checksum of the functions/data and recompute the cache is 
    these values changed

    Parameters
    ----------

    fname : name of the file that will be used to compute the checksum 

    Returns
    -------
    String with a checksum of the function/data 

    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def run_abx(data_file, on, across, by, njobs=1, tmpdir=None, distance=cosine_distance):
    '''`run_abx` wraps ABXpy package, it returnt ABX scores for the tasks on, across and by.
    it can be run in parallel by passing the number of jobs to joblib package.

    This function can cache results, allowing to get fast results for the same input data
    in the features and items files doesn't change (same file checksum/hash)
    and tasks.

    Parameters
    ----------

    data_file : [string]
        name of the item and features files without the extensionn
        these files should exist, and the name of the 
        files are : `{data_file}.features` and `{data_file}.items`

    on : [string]
        the name of the column containing the ON data. It must be 
        exist in featrues and items files

    across : [list]
        a list that contain the ACROSS column names, the elements of the 
        list are strings and need to be present on the  `{data_file}.features` 
        and `{data_file}.items`, it can be empty if this task is not 
        computed

    by : [list]
        a list that contains the BY column names as strings, the names
        must be present in the features and items files or be empty if
        BY task is not been computed

    njobs : [int, default: 1]
        number of jobs passed to joblib to compute ABX 

    tmpdir : [string, default:None] 
        a temporary directory where preliminary data will be stored

    distance : [function, default: cosine_distance]
        a distance function to pass ABXpy, it need 3 arguments x, y and
        normalizer

    item_features_hash  [string, default:'0']
        it is the checksum of the items file and will be used 
        to check if the function was aready used with the passed arguments,
        if it exist a cached version will be used (faster), if not
        the function will be compute the abx scores and keep track of the
        checksum
    
    Returns
    -------
    String with a the ABX results, the result is a csv string with 
    multiple lines (separated by \\n) and fields separated by tabs (\\t)
    '''

    if not distance:
        distance = cosine_distance

    # the checksum is used find diff data within the same same file name ...
    item_file = '{}.item'.format(data_file)
    feature_file = '{}.features'.format(data_file)
    try:
        abx_id = md5(item_file) + md5(feature_file)

    except:
        raise ValueError('cannot compute hash_items_features')

    # I read the results from the cache, if it does't exit it will run the
    # function and update the cache, but it that data exist then it will get
    # those scores from the cache. I am checking also if the result file exist,
    # if not it will be created from the data in the cache

    analyze_data = memorizable_abx(data_file, on, across, by, njobs, tmpdir, distance, abx_id)
    analyze_file = '{}.csv'.format(data_file)
    with open(analyze_file, 'w') as afile:
        afile.write(analyze_data)

    return analyze_data



def abx_by_on(features, labels):
    """Compute ABX scores only for on labels""" 

    labels = [x for x in labels.tolist()]
    features = features.tolist()
    features_by_on = defaultdict(lambda: defaultdict(list))
    features_by_hash = {}

    # to avoid repeated computations and reduce memory I cache features
    n = count()
    for on, feat in zip(labels, features):
        on_ = on[0]
        # I join the hash with a number to track features that are similar (hash) ...
        hash_feat = "{}-{}".format(hash(tuple(feat)), n.next())
        features_by_hash[hash_feat] = feat
        features_by_on[on_][hash_feat].append(feat)

    # precompute distances
    distances = dict()
    distance = lambda a, b: scipy.spatial.distance.cosine(a, b)
    for a, b in permutations(features_by_hash.keys(), 2):
        dist = distance(features_by_hash[a], features_by_hash[b])
        distances[(a,b)] = dist
        distances[(b,a)] = dist
    for k in features_by_hash.keys():
        distances[(k, k)] = distance(features_by_hash[a], features_by_hash[b])

    # compute abx for all triplets ...
    res_abx = {}
    f = features_by_on
    d = distances
    for a_, b_ in permutations(features_by_on.keys(), 2):
        if a_ == b_:
            continue
        
        counts, n = 0.0, 0
        for b, (a, x) in product(features_by_on[b_], combinations(features_by_on[a_], 2)):
            counts += 0.5 if distances[(a, x)] == distances[(b, x)] else int(d[(a, x)] < distances[(b, x)])
            counts += 0.5 if distances[(x, a)] == distances[(b, a)] else int(distances[(x, a)] < distances[(b, a)])
            n += 2
        
        res_abx[(b_, a_)]= [counts, n]

    # and print the results
    for pairs in res_abx.keys():
        a, b = pairs
        counts, n = res_abx[pairs] 
        try:
            abx = float(counts)/n
        except ZeroDivisionError:
            abx = 0.0

        res_line = "{}\t{}\t{:.8f}\t{:d}".format(a, b, abx, n)
        print(res_line)


@memory.cache
def compute_abx(features, labels, on, across=None, by=None, njobs=1,
                distance=cosine_distance, tmpdir=None):
    ''' `compute_abx` computes ABX scores the features and labels. This 
    function wraps all the ABXpy file creation, abx score computing. It needs as an 
    input the raw data (features and labels).


    Parameters
    ----------

    features : [list of np.array]
        numpy arrays that contains the features.  

    labels : [list]
        a python list containing the labels of the features, mapped one by one
        in rows with features

    on : [int]
        the column number in features containing the ON labels, not required
        if the list of labels is given

    across : [list, default: None]
        the column numbers containing the ACROSS task in the features data

    by : [list, default:None]
        the column numbers containing the BY task in the features data

    njobs : [int, default: 1]
        number of jobs passed to joblib to compute ABX 

    distance : [function, default: cosine_distance]
        a distance function to pass ABXpy, it need 3 arguments x, y and
        normalizer
    

    Returns
    -------
    abx_scores: [pandas DataFrame]
        A pandas dataframe that contains the ON (pairs), abx scores and number of triplets
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
    run_abx(data_file, on, across, by, njobs, tmpdir=tmpdir, distance=distance)
    abx_scores = pd.read_csv('{}.csv'.format(data_file), sep='\t')

    # cleaning
    item_file = '{}.item'.format(data_file)
    feature_file = '{}.features'.format(data_file)
    distance_file = '{}.distance'.format(data_file)
    score_file = '{}.score'.format(data_file)
    task_file = '{}.abx'.format(data_file)
    analyze_file = '{}.csv'.format(data_file)
    remove_files = [distance_file, score_file, task_file, analyze_file]
    remove_files += [item_file, feature_file]
    map(os.remove, filter(os.path.exists, remove_files))

    return abx_scores


if __name__ == "__main__":
    import doctest
    doctest.testmod()
