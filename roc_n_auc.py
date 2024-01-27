#! /usr/bin/env python3

import sys
import argparse

import pandas as pd
import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser(description='Save ROC and AUC')

parser.add_argument('results_file', type=argparse.FileType('r'), help='results file')
parser.add_argument('-p', '--output_prefix', help='prefix for output files')

arguments = parser.parse_args(sys.argv[1:])

# print(arguments.results_file.name)

results = pd.read_pickle(arguments.results_file.name)

fpr, tpr, _ = sklearn.metrics.roc_curve(results['actual'], results['prediction'])

auc = sklearn.metrics.auc(fpr, tpr)

np.savetxt(arguments.output_prefix + 'ROC.txt', np.c_[fpr, tpr])
np.savetxt(arguments.output_prefix + 'AUC.txt', np.array([auc]))