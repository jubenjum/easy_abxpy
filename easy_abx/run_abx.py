#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" 
run_abx.py: compute ABX score using the package from 

https://github.com/bootphon/ABXpy

"""

import sys

try:
    # python 3.6
    from pathlib import Path
except ImportError:
    # python 2.6
    from pathlib2 import Path

from scipy.spatial.distance import euclidean

from utils import run_abx


def main():
    import argparse

    parser = argparse.ArgumentParser( prog='run_abx.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='compute ABX score using package from https://github.com/bootphon/ABXpy')

    parser.add_argument('abx_experiment_name', nargs=1,
	help=('The experiment name, or the file name stripped the suffix, '
              'all names will contain this name.'))

    parser.add_argument('--on', required=True,
                        help=('The name of the ON labels, '
                            'it is mandatory and is an quoted string\n'
                            'for exaple --on "on_column_name"'))

    parser.add_argument('--across', nargs='+',
                        help=('The name(s) of the ACROSS features, '
                            ' can be zero or more cuoted names separated by spaces \n'
                            ' for exaple: --across "field1" "field2"'))

    parser.add_argument('--by', nargs='+',
                        help='The name(s) of the BY labels, same format than ACROSS')

    parser.add_argument('--tmpdir', required=False,
                        help='temporary directory name')

    parser.add_argument('--distance', required=False,
                        help=('uses the distnace function defined on the' 
                              ' distance file given on this argument.'
                              ' default distance function will be '))

    parser.add_argument('-j', '--njobs', default=1, type=int, 
                        help='run ABXpy in j parallel jobs')

    args = parser.parse_args()
    abx_name = args.abx_experiment_name[0]
    if args.distance:
        distance_file = Path(args.distance)
        if distance_file.is_file():
            exec(distance_file.read_text())
        else:
            print("ERROR: '{}' doesn't exist".format(args.distance))
            sys.exit()
        if not "distance" in dir():
            print(dir())
            print("No function 'distance' in file {}".format(args.distance))
            sys.exit()
    else:
        distance = None
    
    run_abx(abx_name, args.on, args.across, args.by, args.njobs, args.tmpdir, distance=distance)



if __name__ == '__main__':
    main()

