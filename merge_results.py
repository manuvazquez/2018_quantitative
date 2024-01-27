#! /usr/bin/env python3

import os
import sys
import glob
import argparse
import json
import functools

import pandas as pd
import numpy as np
import sklearn.metrics
import scipy

import data_processing
import evaluation

import manu.util

parser = argparse.ArgumentParser(description='merge results')

parser.add_argument('input_path', default=os.getcwd(), action=manu.util.ReadableDir)
parser.add_argument('-o', '--output_path', default=os.getcwd(), action=manu.util.ReadableDir)

arguments = parser.parse_args(sys.argv[1:])

# ------------

results_file = 'results.json'
predictions_file = 'predictions.pickle'
output_sensitivity = 'sensitivity_RNN.txt'
output_auc = 'AUC_RNN.txt'
output_roc_prefix = 'ROC_'

architectures = [
	'simple_with_time', 'he4_with_time', 'gly_with_time', 'ca125_he4_with_time', 'ca125_gly_with_time',
	'ca125_he4_gly_with_time'
]

collected_folds_directories = ['5_folds', '10_folds']

desired_specificity = 0.9

# ------------


def average_roc(fpr_list, tpr_list):

	n = len(fpr_list)

	# the union of all the FPRs
	overall_fpr = functools.reduce(np.union1d, fpr_list)

	# memory is reserved for the corresponding TPRs
	overall_tpr = [None] * len(overall_fpr)

	for i_fpr_value, fpr_value in enumerate(overall_fpr):

		overall_tpr[i_fpr_value] = 0.0

		for fpr, tpr in zip(fpr_list, tpr_list):

			# the TPR for any arbitrary FPR is estimated by **interpolation**
			overall_tpr[i_fpr_value] += scipy.interp(fpr_value, fpr, tpr)

		overall_tpr[i_fpr_value] /= n

	return overall_fpr, overall_tpr


architectures_to_names = {
	'simple_with_time': 'CA125',
	'HE4_with_time': 'HE4',
	'he4_with_time': 'HE4',
	'HE4': 'HE4 (no time)',
	'ca125_he4_gly_with_time': 'CA125+HE4+Gly',
	'ca125_he4_gly': 'CA125+HE4+Gly (no time)',
	'Gly_with_time': 'Gly',
	'gly_with_time': 'Gly',
	'Gly': 'Gly (no time)',
	'simple': 'CA125 (no time)',
	'ca125': 'CA125 (no time)',
	'ca125_he4': "CA125+HE4 (no time)",
	'ca125_he4_with_time': 'CA125+HE4',
	'ca125_gly': "CA125-Gly (no time)",
	'ca125_gly_with_time': "CA125+Gly"
}

architectures_to_filenames = {
	'simple_with_time': 'CA125',
	'HE4_with_time': 'HE4',
	'he4_with_time': 'HE4',
	'ca125_he4_gly_with_time': 'CA125_HE4_Gly',
	'Gly_with_time': 'Gly',
	'gly_with_time': 'Gly',
	'ca125_he4_with_time': 'CA125_HE4',
	'ca125_gly_with_time': "CA125_Gly"
}

# ------------


summary = {}

# for every algorithm/architecture
for arch in architectures:

	# a full dictionary for every architecture
	summary[arch] = {}

	arch_dir = os.path.join(arguments.input_path, arch)

	across_fold_collections_fpr = []
	across_fold_collections_tpr = []

	arch_sensitivities = []
	arch_auc = []

	# for every collection of folds (5 folds, 10 folds...)
	for fold_collection_dir in collected_folds_directories:

		summary[arch][fold_collection_dir] = {}

		dir = os.path.join(arch_dir, fold_collection_dir)

		sensitivities = []
		auc = []

		fold_collection_fpr = []
		fold_collection_tpr = []

		# an empty dictionary to store the results for the "unfolded" data
		summary[arch][fold_collection_dir]['unfolded'] = {}

		# a Pandas dataframe for storing the predictions for the *entire* dataset (all the folds)
		summary[arch][fold_collection_dir]['unfolded']['results'] = pd.DataFrame()

		# for every individual fold
		for d in glob.glob(os.path.join(dir, 'fold_?')):

			# =============== sensitivity and AUC

			with open(os.path.join(d, results_file)) as f:

				res = json.load(f)

			sensitivities.append(res["sensitivity"])
			auc.append(res["roc_auc"])

			# =============== ROC

			# result for every sample along with the true label
			results = pd.read_pickle(os.path.join(d, predictions_file))

			# the results for this fold are stacked into the corresponding dataframe
			summary[arch][fold_collection_dir]['unfolded']['results'] = summary[arch][fold_collection_dir]['unfolded'][
				'results'].append(pd.read_pickle(os.path.join(d, predictions_file)))

			# FPR and TPR are computed...
			fpr, tpr, _ = sklearn.metrics.roc_curve(results['actual'], results['prediction'])

			# we want a vector of FPRs without duplicates
			unique_fpr, i_first_occur, n_occur = np.unique(fpr, return_counts=True, return_index=True)

			# for every FPR, we pick the best TPR...
			unique_tpr = [None] * len(unique_fpr)
			for i_unique_fpr in range(len(unique_fpr)):

				# ...that corresponding to the last occurrence
				unique_tpr[i_unique_fpr] = tpr[i_first_occur[i_unique_fpr] + n_occur[i_unique_fpr] - 1]

			# ...and stored
			fold_collection_fpr.append(unique_fpr)
			fold_collection_tpr.append(unique_tpr)

		# =============== sensitivity and AUC

		summary[arch][fold_collection_dir]['sensitivity'] = np.mean(sensitivities)
		summary[arch][fold_collection_dir]['AUC'] = np.mean(auc)

		arch_sensitivities.append(summary[arch][fold_collection_dir]['sensitivity'])
		arch_auc.append(summary[arch][fold_collection_dir]['AUC'])

		# =============== ROC

		overall_fpr_values, overall_tpr_values = average_roc(fold_collection_fpr, fold_collection_tpr)

		summary[arch][fold_collection_dir]['FPR'] = overall_fpr_values
		summary[arch][fold_collection_dir]['TPR'] = overall_tpr_values

		# stored for averaging ROCs across "fold_collection"s
		across_fold_collections_fpr.append(overall_fpr_values)
		across_fold_collections_tpr.append(overall_tpr_values)

		# ------

		# for the sake of readability
		arch_results = summary[arch][fold_collection_dir]['unfolded']['results']

		binary_output_results = arch_results.copy()
		binary_output_results['actual'] = binary_output_results['actual'].astype(int)
		binary_output_results.to_csv(
			os.path.join(dir, os.path.splitext(predictions_file)[0] + '.txt'), header=False, index=False)

		# the threshold required for a given specificity...
		threshold = data_processing.fixed_specificity_threshold(arch_results, desired_specificity)

		# ...and the implied sensitivity,...
		sensitivity = evaluation.sensitivity(arch_results, threshold)

		# ...ROC,...
		fpr, tpr, _ = sklearn.metrics.roc_curve(arch_results['actual'], arch_results['prediction'])

		# ..., and AUC
		auc = sklearn.metrics.auc(fpr, tpr)

		# results are stored
		summary[arch][fold_collection_dir]['unfolded']['sensitivity'] = sensitivity
		summary[arch][fold_collection_dir]['unfolded']['FPR'] = fpr
		summary[arch][fold_collection_dir]['unfolded']['TPR'] = tpr
		summary[arch][fold_collection_dir]['unfolded']['AUC'] = auc

		print(f'sensitivity for {arch} is {sensitivity}')

	overall_fpr_values, overall_tpr_values = average_roc(across_fold_collections_fpr, across_fold_collections_tpr)
	summary[arch]['FPR'] = overall_fpr_values
	summary[arch]['TPR'] = overall_tpr_values

	summary[arch]['sensitivity'] = np.mean(arch_sensitivities)
	summary[arch]['AUC'] = np.mean(arch_auc)

print(summary)

# =============== sensitivity and AUC

names, sensitivity_values, auc_values = [], [], []

for arch in architectures:

	names.append(architectures_to_names[arch])
	sensitivity_values.append(summary[arch]['sensitivity'])
	auc_values.append(summary[arch]['AUC'])

	roc_filename = output_roc_prefix + architectures_to_filenames[arch] + '.txt'

	with open(os.path.join(arguments.output_path, roc_filename), 'wb') as f:

		np.savetxt(f, np.c_[summary[arch]['FPR'], summary[arch]['TPR']], fmt='%s', delimiter='\t')

with open(os.path.join(arguments.output_path, output_sensitivity), 'wb') as f:

	np.savetxt(f, np.c_[names, sensitivity_values], fmt='%s', delimiter='\t')

with open(os.path.join(arguments.output_path, output_auc), 'wb') as f:

	# np.savetxt(f, np.c_[names, auc_values], fmt='%s', delimiter='\t')

	np.savetxt(f, np.array(names).reshape(1, -1), fmt='%s', delimiter='\t')
	np.savetxt(f, np.array(auc_values).reshape(1, -1))

# =============== unfolded data

for fold_collection_dir in collected_folds_directories:

	filename_prefix = 'unfolded_' + fold_collection_dir + '_'

	names, sensitivity_values, auc_values = [], [], []

	for arch in architectures:

		# for the sake of readability
		res = summary[arch][fold_collection_dir]['unfolded']

		names.append(architectures_to_names[arch])
		sensitivity_values.append(res['sensitivity'])
		auc_values.append(res['AUC'])

		roc_filename = filename_prefix + output_roc_prefix + architectures_to_filenames[arch] + '.txt'

		with open(os.path.join(arguments.output_path, roc_filename), 'wb') as f:

			np.savetxt(f, np.c_[res['FPR'], res['TPR']], fmt='%s', delimiter='\t')

	with open(os.path.join(arguments.output_path, filename_prefix + output_sensitivity), 'wb') as f:

		np.savetxt(f, np.c_[names, sensitivity_values], fmt='%s', delimiter='\t')

	with open(os.path.join(arguments.output_path, filename_prefix + output_auc), 'wb') as f:

		np.savetxt(f, np.array(names).reshape(1, -1), fmt='%s', delimiter='\t')
		np.savetxt(f, np.array(auc_values).reshape(1, -1))