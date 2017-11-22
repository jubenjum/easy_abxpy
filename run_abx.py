#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import sys
package_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not(package_path in sys.path):
    sys.path.append(package_path)


import ABXpy.task
import ABXpy.distances.distances  
import ABXpy.distances.distances as distances
import ABXpy.distances.metrics.cosine as cosine
import ABXpy.distances.metrics.dtw as dtw
import ABXpy.score as score
import ABXpy.misc.items as items
import ABXpy.analyze as analyze

import scipy.spatial.distance

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
def cosine_distance(x, y, normalized):
    return scipy.spatial.distance.cosine(x, y)
    
def correlation_distance(x, y, normalized):
    return scipy.spatial.distance.correlation(x, y)

def dtw_cosine_distance(x, y, normalized):
    return dtw.dtw(x, y, cosine.cosine_distance, normalized)


def run_abx(data_file, on, across, by, njobs, distance=cosine_distance):
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
    task = ABXpy.task.Task(item_file, on, across=across, by=by)
    task.generate_triplets(taskfilename)
    distances.compute_distances(feature_file, '/features/', taskfilename,
                                distance_file, distance, normalized=True, n_cpu=njobs)
    score.score(taskfilename, distance_file, scorefilename)
    analyze.analyze(taskfilename, scorefilename, analyzefilename)


def main():
    import argparse

    parser = argparse.ArgumentParser( prog='run_abx.py',
            formatter_class=argparse.RawDescriptionHelpFormatter, 
            description='compute ABX score')

    parser.add_argument('abx_experiment_name', nargs=1, 
	help='The experiment name, or the file name stripped the suffix')
    
    parser.add_argument('--on', required=True, help='on label "STRING"')
    
    parser.add_argument('--across',  help='across feature "STRING"')
    
    parser.add_argument('--by', help='by label "STRING"')
    
    parser.add_argument('-j', '--njobs', default=1, 
            type=int, help='run ABXpy in j parallel jobs')

    args = parser.parse_args()
    abx_name = args.abx_experiment_name[0]
    run_abx(abx_name, args.on, args.across, args.by, args.njobs)


if __name__ == '__main__':
    main()
















