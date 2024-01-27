#! /usr/bin/env python3

import os
import sys
import argparse

import pandas as pd
import numpy as np

import manu.util

parser = argparse.ArgumentParser(description='merge results')

parser.add_argument('input_path', default=os.getcwd(), nargs='?', action=manu.util.ReadableDir)
parser.add_argument('-o', '--output_path', default=os.getcwd(), action=manu.util.ReadableDir)

arguments = parser.parse_args(sys.argv[1:])

# ------------

minimum_specificity = 0.9
output_file = 'sensitivity_BCP.txt'
markers_list = ['CA125', 'HE4', 'Gly', 'CA125+HE4', 'CA125+Gly', 'CA125+HE4+Gly']

# ------------


data = pd.read_csv(os.path.join(arguments.input_path, 'ROC_curve.txt'), sep='\t', header=None,
                   names=['threshold', 'CA125 specificity', 'HE4 specificity', 'Gly specificity',
                            'CA125+HE4 specificity', 'CA125+Gly specificity', 'CA125+HE4+Gly specificity',
                            'CA125 sensitivity', 'HE4 sensitivity', 'Gly sensitivity',
                            'CA125+HE4 sensitivity', 'CA125+Gly sensitivity', 'CA125+HE4+Gly sensitivity'
                            ]
                   )

sensitivities = []

# for marker in ['CA125']:
for marker in markers_list:

	# boolean stating whether the required specificity is attained
	at_least_given_specificity = data[marker + ' specificity'] >= minimum_specificity

	# the sensitivity corresponding to the *first* element satisfying the condition (it should maximize the sensitivity)
	# is appended to the list
	sensitivities.append(data[at_least_given_specificity].iloc[0][marker + ' sensitivity'])

	print(f'sensitivity for {marker}: {sensitivities[-1]}')


with open(os.path.join(arguments.output_path, output_file), 'wb') as f:

	np.savetxt(f, np.c_[markers_list, sensitivities], fmt='%s', delimiter='\t')
