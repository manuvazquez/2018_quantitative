#! /usr/bin/env python3

import os

import numpy as np
import sklearn.model_selection
import pandas as pd

import data_processing

# ------------------ parameters

# the fraction of the dataset used for training
training_percentage = 0.5

# path where to look for the files specified below
data_directory = '/home/manu/rnns/datos_bd'

# the full dataset will be saved into a pickle file
full_data_file = 'raw_data.pickle'

# the names of the files containing, respectively, the healthy and ill subjects data
CA125_healthy_subjects_file = 'BD_control_sin.txt'
CA125_ill_subjects_file = 'BD_sin.txt'

Gly_healthy_subjects_file = 'BD_Gly_control_sin.txt'
Gly_ill_subjects_file = 'BD_Gly_sin.txt'

HE4_healthy_subjects_file = 'BD_HE4_control_sin.txt'
HE4_ill_subjects_file = 'BD_HE4_sin.txt'

# ------------------

CA125_healthy_subjects_file = os.path.join(data_directory, CA125_healthy_subjects_file)
CA125_ill_subjects_file = os.path.join(data_directory, CA125_ill_subjects_file)

Gly_healthy_subjects_file = os.path.join(data_directory, Gly_healthy_subjects_file)
Gly_ill_subjects_file = os.path.join(data_directory, Gly_ill_subjects_file)

HE4_healthy_subjects_file = os.path.join(data_directory, HE4_healthy_subjects_file)
HE4_ill_subjects_file = os.path.join(data_directory, HE4_ill_subjects_file)

# for the sake of reproducibility
np.random.seed(7)

# data is not "curated" (no data column is dropped)
CA125_raw_data = data_processing.read_and_clean(
	healthy=CA125_healthy_subjects_file, ill=CA125_ill_subjects_file, curate=False,
	measurements_columns=['CA125_{}'.format(i) for i in range(1, 6)])

Gly_raw_data = data_processing.read_and_clean(
	healthy=Gly_healthy_subjects_file, ill=Gly_ill_subjects_file, curate=False,
	measurements_columns=['Gly_{}'.format(i) for i in range(1, 6)])

HE4_raw_data = data_processing.read_and_clean(
	healthy=HE4_healthy_subjects_file, ill=HE4_ill_subjects_file, curate=False,
	measurements_columns=['HE4_{}'.format(i) for i in range(1, 6)])

raw_data = pd.concat(
	[CA125_raw_data, Gly_raw_data.filter(regex='Gly_[1-5]'), HE4_raw_data.filter(regex='HE4_[1-5]')],
	axis=1)

# dataframe is saved as is
raw_data.to_pickle(full_data_file)

# splitting intro training and test
raw_train_df, raw_test_df = sklearn.model_selection.train_test_split(
	raw_data, train_size=training_percentage, stratify=raw_data['ill'])

ill = raw_train_df['ill']
healthy = ~ill

# ill subjects from the *training* set are saved to a csv file in the appropriate format...
raw_train_df[ill].drop('ill', axis=1).to_csv(
	'CA125_train_ill.csv', header=False, index=True, sep=' ', columns=CA125_raw_data.columns)
raw_train_df[ill].drop('ill', axis=1).to_csv(
	'Gly_train_ill.csv', header=False, index=True, sep=' ', columns=Gly_raw_data.columns)
raw_train_df[ill].drop('ill', axis=1).to_csv(
	'HE4_train_ill.csv', header=False, index=True, sep=' ', columns=HE4_raw_data.columns)

# ...the same for healthy subjects
raw_train_df[healthy].drop('ill', axis=1).to_csv(
	'CA125_train_healthy.csv', header=False, index=True, sep=' ', columns=CA125_raw_data.columns)
raw_train_df[healthy].drop('ill', axis=1).to_csv(
	'Gly_train_healthy.csv', header=False, index=True, sep=' ', columns=Gly_raw_data.columns)
raw_train_df[healthy].drop('ill', axis=1).to_csv(
	'HE4_train_healthy.csv', header=False, index=True, sep=' ', columns=HE4_raw_data.columns)

# ------

ill = raw_test_df['ill']
healthy = ~ill

# just like above, ill and healthy subjects, now from the *test* set, are dumped into csv files
raw_test_df[ill].drop('ill', axis=1).to_csv(
	'CA125_test_ill.csv', header=False, index=True, sep=' ', columns=CA125_raw_data.columns)
raw_test_df[ill].drop('ill', axis=1).to_csv(
	'Gly_test_ill.csv', header=False, index=True, sep=' ', columns=Gly_raw_data.columns)
raw_test_df[ill].drop('ill', axis=1).to_csv(
	'HE4_test_ill.csv', header=False, index=True, sep=' ', columns=HE4_raw_data.columns)

raw_test_df[healthy].drop('ill', axis=1).to_csv(
	'CA125_test_healthy.csv', header=False, index=True, sep=' ', columns=CA125_raw_data.columns)
raw_test_df[healthy].drop('ill', axis=1).to_csv(
	'Gly_test_healthy.csv', header=False, index=True, sep=' ', columns=Gly_raw_data.columns)
raw_test_df[healthy].drop('ill', axis=1).to_csv(
	'HE4_test_healthy.csv', header=False, index=True, sep=' ', columns=HE4_raw_data.columns)
