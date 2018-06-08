#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
            description='compute ABX score')

    parser.add_argument('abx_experiment_name', nargs=1,
	help='The experiment name, or the file name stripped the suffix')

    parser.add_argument('--on', required=True,
                        help='on labels "STRING"')

    parser.add_argument('--across', nargs='+',
                        help='across features zero or more "STRING"')

    parser.add_argument('--by', nargs='+',
                        help='by labels zero or more "STRINGS"')

    parser.add_argument('--tmpdir', required=False,
                        help='temporary directory "STRING"')

    parser.add_argument('--distance', required=False,
                        help=('uses the distnace function defined on the' 
                              ' distance file given on this argument.'
                              ' default distance function will be '))

    parser.add_argument('-j', '--njobs', default=1,
            type=int, help='run ABXpy in j parallel jobs')

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

