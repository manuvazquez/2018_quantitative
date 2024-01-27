#! /usr/bin/env python3

import os
import json

import sklearn.model_selection

import data_processing

# ------------------ parameters

with open('parameters.json') as json_data:

	# the parameters file is read to memory
	parameters = json.load(json_data)

# the fraction of the dataset used for training (only meaningful for `StratifiedShuffleSplit`)
training_percentage = 0.9

n_splits = 10

output_directory_prefix = 'fold_'

# whether or not to discard subjects with a single measurement
# dismiss_single_measurements = True
dismiss_single_measurements = False

# whether or not cut off measurements off control subjects so that percentage of subjects with each number of
# measurements matches between control and cases datasets
try_matching_n_measurements_percentages = False

# ---

# path where to look for the files specified below
data_directory = '/home/manu/rnns/datos_bd'

# the names of the files containing, respectively, the healthy and ill subjects data
CA125_healthy_subjects_file = 'BD_control_sin.txt'
CA125_ill_subjects_file = 'BD_sin.txt'

Gly_healthy_subjects_file = 'BD_Gly_control_sin.txt'
Gly_ill_subjects_file = 'BD_Gly_sin.txt'

HE4_healthy_subjects_file = 'BD_HE4_control_sin.txt'
HE4_ill_subjects_file = 'BD_HE4_sin.txt'

# ------------------

# the data directory is prepended to every file

CA125_healthy_subjects_file = os.path.join(data_directory, CA125_healthy_subjects_file)
CA125_ill_subjects_file = os.path.join(data_directory, CA125_ill_subjects_file)

Gly_healthy_subjects_file = os.path.join(data_directory, Gly_healthy_subjects_file)
Gly_ill_subjects_file = os.path.join(data_directory, Gly_ill_subjects_file)

HE4_healthy_subjects_file = os.path.join(data_directory, HE4_healthy_subjects_file)
HE4_ill_subjects_file = os.path.join(data_directory, HE4_ill_subjects_file)

# ------------------

# all the data is read into a (single) dataframe
# NOTE: data is not "curated", i.e., no data column is dropped
data, metrics_columns = data_processing.read_and_clean(parameters['data'], [
	(CA125_healthy_subjects_file, CA125_ill_subjects_file, 'CA125'),
	(HE4_healthy_subjects_file, HE4_ill_subjects_file, 'HE4'),
	(Gly_healthy_subjects_file, Gly_ill_subjects_file, 'Gly')
], curate=False)

if dismiss_single_measurements:

	# only those subjects with more than one measurement are kept
	data = data[data['n_measurements'] > 1]


if try_matching_n_measurements_percentages:

	data['modified'] = False

	max_n_measurements = data['n_measurements'].max()

	# for n_measurements in range(1, max_n_measurements + 1):
	for n_measurements in range(1, max_n_measurements):

		# print(n_measurements)

		# for the sake of convenience
		ill = data[data['ill']]
		healthy = data[~data['ill']]

		# number of *ill* subjects with (exactly) this number of measurements...
		fraction_ill_subjects = (ill['n_measurements'] == n_measurements).sum() / len(ill)

		# ...and of healthy subjects
		fraction_healthy_subjects = (healthy['n_measurements'] == n_measurements).sum() / len(healthy)

		# fraction and number of healthy subjects to be set
		fraction_healthy_subjects_to_set = fraction_ill_subjects - fraction_healthy_subjects
		n_healthy_subjects_to_set = int(fraction_healthy_subjects_to_set * len(healthy))

		# if we have more healthy subjects with this number of measurements
		if n_healthy_subjects_to_set < 0:

			break

		# print(n_healthy_subjects_to_set)

		# in principle, we cut off measurements from the subjects with the largest number of measurements
		n_measurements_sought = max_n_measurements

		index = healthy[(healthy['n_measurements'] == n_measurements_sought) & ~healthy['modified']].sample(
			n=n_healthy_subjects_to_set).index

		data.loc[index, 'n_measurements'] = n_measurements
		data.loc[index, 'modified'] = True

	# for the sake of convenience
	ill = data[data['ill']]
	healthy = data[~data['ill']]

	print(ill.groupby('n_measurements').size() / len(ill))
	print(healthy.groupby('n_measurements').size() / len(healthy))

# ------------------

# NOTE: `random_state` is set for the sake of reproducibility

# cross_validator = sklearn.model_selection.StratifiedShuffleSplit(
# 	n_splits=n_splits, train_size= training_percentage, test_size=1. - training_percentage, random_state=7)

cross_validator = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, random_state=7)

# cross_validator = sklearn.model_selection.LeaveOneOut()

for i_split, (i_train, i_test) in enumerate(cross_validator.split(data, data['ill'])):

	output_directory = output_directory_prefix + str(i_split)
	os.makedirs(output_directory, exist_ok=True)

	train_df, test_df = data.iloc[i_train], data.iloc[i_test]

	# ----- training

	ill = train_df['ill']
	healthy = ~ill

	for filename, variable in zip(
			['CA125_train_ill.csv', 'Gly_train_ill.csv', 'HE4_train_ill.csv'], metrics_columns):

		train_df[ill].drop('ill', axis=1).to_csv(
			os.path.join(output_directory, filename), header=False, index=True, sep=' ', columns=variable)

	for filename, variable in zip(
			['CA125_train_healthy.csv', 'Gly_train_healthy.csv', 'HE4_train_healthy.csv'], metrics_columns):

		train_df[healthy].drop('ill', axis=1).to_csv(
			os.path.join(output_directory, filename), header=False, index=True, sep=' ', columns=variable)

	# ----- test

	ill = test_df['ill']
	healthy = ~ill

	for filename, variable in zip(
			['CA125_test_ill.csv', 'Gly_test_ill.csv', 'HE4_test_ill.csv'], metrics_columns):

		test_df[ill].drop('ill', axis=1).to_csv(
			os.path.join(output_directory, filename), header=False, index=True, sep=' ', columns=variable)

	for filename, variable in zip(
			['CA125_test_healthy.csv', 'Gly_test_healthy.csv', 'HE4_test_healthy.csv'], metrics_columns):

		test_df[healthy].drop('ill', axis=1).to_csv(
			os.path.join(output_directory, filename), header=False, index=True, sep=' ', columns=variable)
