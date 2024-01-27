from typing import Tuple, Sequence, List
import copy

import numpy as np
import pandas as pd
import sklearn.metrics

import utils.geometry

# a measurement equal to this value is ignored
not_a_measurement_value = 0.0


def rotate(row, t_columns, measurements_columns, degrees: float):

	# it only makes sense rotating the signal if there is more than one measurement
	if row['n_measurements'] > 1:

		t = row[t_columns[:row['n_measurements']]].values
		s = row[measurements_columns[:row['n_measurements']]].values

		new_t, new_s = utils.geometry.rotate_signal(t, s, degrees)

		row[t_columns[:row['n_measurements']]] = new_t
		row[measurements_columns[:row['n_measurements']]] = new_s

	return row


def shift(row, measurements_columns, value: float):

	row[measurements_columns[:row['n_measurements']]] += value

	return row


def fill_missing_values(
		row, time_columns, measurements_columns, max_measurements: int, measurement_filling, time_filling):
	"""
	It fills the missing values, according to the column "n_measurements" and `max_measurements`,  with `filling`.
	This function is meant to be used as argument to Pandas' "apply"

	Parameters
	----------
	row : Pandas series
	time_columns: list of strings
		A list specifying the columns containing the time stamps associated with the measurements.
	measurements_columns: list of strings
		A list specifying the columns the measurements.
	max_measurements : int
	measurement_filling : float or NaN
	time_filling: float or NaN

	Returns
	-------
	row : Pandas series
		The modified input series.

	"""

	# if the number of measurements in this row, according to the corresponding field, is below the maximum...
	if row['n_measurements'] < max_measurements:

		# ...measurement after the last one, "int(row['n_measurements'])", are filled in with "filling"
		row[measurements_columns[int(row['n_measurements']):]] = measurement_filling

		# ...and the same for the corresponding time instants
		row[time_columns[int(row['n_measurements']):]] = time_filling

	return row


def read_and_clean(
		data_parameters: dict, biomarkers_paths: Tuple, curate: bool = True) -> Tuple[pd.DataFrame, List[List]]:

	data_format = data_parameters["format"]

	# parameters to be passed directly to Pandas' `read_csv`
	read_csv_arguments = data_parameters[data_format]["Pandas' read_csv"]

	# columns to be read, excluding those with measurements (a list of lists)
	base_columns = data_parameters[data_format]["columns"]

	# maximum number of measurements
	n_measurements = data_parameters[data_format]["maximum number of measurements"]

	i_measurements_columns = data_parameters[data_format]["index measurement columns"]
	i_time_columns = data_parameters[data_format]["index time columns"]

	# index for the column which yields the "label" of each subject
	i_index_column = data_parameters[data_format]["index column"]

	# these columns are dropped at the end
	columns_to_be_dropped = data_parameters[data_format]["columns to be dropped"]

	# names for the time columns
	t_columns = ['t_{}'.format(i) for i in range(1, 1 + n_measurements)]

	measurements_columns_list = []

	healthy_subjects_list = []
	ill_subjects_list = []

	# a list of lists, each one with all the columns needed for dealing with every metric on its own (hence, there is
	# overlapping among the lists)
	metrics_columns = []

	is_first_metric = True

	# for every metric, a tuple with
	# i)   the file name for the healthy subjects
	# ii)  the file name for the ill subjects
	# iii) the metric name
	for healthy, ill, metric_name in biomarkers_paths:

		# we want an identical copy of `base_columns` in each iteration
		columns = copy.deepcopy(base_columns)

		# the prefix used in the names of the columns for this metric
		metric_prefix = data_parameters[metric_name]["columns prefix"]

		# a list with the names of the measurement columns
		measurements_columns = [f'{metric_prefix}{i}' for i in range(1, 1 + n_measurements)]

		# this is later on used when "curating" the data
		measurements_columns_list.append(measurements_columns)

		# the "measurement" and "time" columns (lists) are inserted at the appropriate places in the overall list of
		# columns
		columns.insert(i_measurements_columns, measurements_columns)
		columns.insert(i_time_columns, t_columns)

		# print(columns)

		# the list is flattened
		columns = sum(columns, [])

		# the file with the healthy subjects is read
		healthy_subjects = pd.read_csv(healthy, names=columns, index_col=i_index_column, **read_csv_arguments)

		# the *whole* dataframe is added to the list if this is the first metric; otherwise only the measurements
		healthy_subjects_list.append(healthy_subjects if is_first_metric else healthy_subjects[measurements_columns])

		ill_subjects = pd.read_csv(ill, names=columns, index_col=i_index_column, **read_csv_arguments)
		ill_subjects_list.append(ill_subjects if is_first_metric else ill_subjects[measurements_columns])

		metrics_columns.append(list(healthy_subjects.columns))

		is_first_metric = False

	# the dataframes for all of the metrics are *horizontally* concatenated
	healthy_subjects = pd.concat(healthy_subjects_list, axis=1)
	ill_subjects = pd.concat(ill_subjects_list, axis=1)

	# a new column specifying whether the subject is ill or not is added prior to...
	healthy_subjects['ill'] = False
	ill_subjects['ill'] = True

	# ...*vertical* concatenation
	data = pd.concat([healthy_subjects, ill_subjects])

	# data "cleaning"
	if curate:

		# the maximum number of measurements available in any example
		max_measurements = data['n_measurements'].max()

		# useless columns are dropped
		data = data.drop(columns_to_be_dropped, axis=1)

		for measurements_columns in measurements_columns_list:

			# missing values are "flagged"
			data = data.apply(
				fill_missing_values, axis=1, time_columns=t_columns, measurements_columns=measurements_columns,
				max_measurements=max_measurements, measurement_filling=np.nan, time_filling=np.nan)

		# *every* column "dtype" is cast to float by "apply", but some of them'd rather be integers...
		data['n_measurements'] = data['n_measurements'].astype(int)

		# ...or boolean
		data['ill'] = data['ill'].astype(bool)

		# healthy subjects' diagnosis time is meaningless
		data.loc[data.ill==False, 't_diagnosis'] = np.nan

	return data, metrics_columns


def each_subject_last_measurement(df, columns):

	# the label (name of the column) of each subject's last measurement
	last_measurements_labels = pd.Series(np.array(columns)[df['n_measurements'] - 1])

	# the values in df.index are used to select a row and those in last_measurements_labels to select a column
	return pd.Series(df.lookup(df.index, last_measurements_labels), index=df.index)


def each_subject_max_measurement(df, columns):

	return df[columns].max(axis=1)


def fixed_specificity_threshold(results, requested_specificity):

	# 1-specificity, sensitivity and the corresponding thresholds are computed
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(results['actual'], results['prediction'])

	# for the sake of convenience
	specificity = 1. - fpr

	# the index of the *smallest* threshold (higher sensitivity) where specificity is greater than requested
	i_threshold = np.where(specificity > requested_specificity)[0][-1]

	return thresholds[i_threshold]


def clear_measurements_after(n, data, time_columns, measurements_columns):

	data[measurements_columns[n:] + time_columns[n:]] = not_a_measurement_value


def set_last_measurement_according_to_diagnosis(
		data, time_columns, measurements_columns, n_measurements, less_than=2, more_than=1, verbose=True):

	res = data.copy()

	# for the sake of clarity/cleanliness
	res_ill = res[res['ill']]

	# a numpy array with the time until diagnosis from each sample
	t_until_diagnosis = res_ill['t_diagnosis'].values[:, np.newaxis] - res_ill[time_columns].values

	# we only care about samples that are not further away from diagnosis than `less_than` and not closer than
	# `more_than` to diagnosis
	conditions_hold = (t_until_diagnosis < less_than) & (t_until_diagnosis > more_than)

	# true for the rows corresponding to subjects that meet the constraint
	eligible = np.any(conditions_hold, axis=1)

	# subjects that do not meet the above constraint are not accounted for
	conditions_hold = conditions_hold[eligible, :]

	# "hack" to find the *latest* measurement meeting the constraint (only for eligible subjects)
	i_latest_fitting_measurement = np.argmax(conditions_hold.astype(float) * np.arange(1, n_measurements + 1), axis=1)

	# useful for accessing the Pandas dataframe
	index_eligible = res_ill[eligible].index

	# for the eligible subjects with their corresponding latest fitting measurement...
	for index, i_measurement in zip(index_eligible, i_latest_fitting_measurement):

		if verbose:

			n_dropped_measurements = res.loc[index, 'n_measurements'] - (i_measurement + 1)

			print('dropped the {} last measurements of {}'.format(n_dropped_measurements, index))

		# measurements above the latest fitting measurement are flagged
		res.loc[index, measurements_columns[i_measurement+1:]] = not_a_measurement_value

	# indexes of the subjects that are not "eligible"
	index_not_eligible = res_ill[~eligible].index

	# for subjects that don't meet the constraint...
	for index in index_not_eligible:

		res.drop(index, axis=0, inplace=True)

	if verbose:

		print('dropped subjects {}'.format(index_not_eligible.values))

	return res


def add_slopes(data, time_columns, measurements_columns, slope_columns):

	measurements_diffs = data.loc[:, measurements_columns].diff(axis=1)
	time_diffs = data.loc[:, time_columns].diff(axis=1)

	# columns in "time_diffs" are renamed so that they can be operated with those in "measurements_diffs"
	res = measurements_diffs / time_diffs.rename(columns=dict(zip(time_columns, measurements_columns)))

	# the first slop is assumed to be zero
	res[measurements_columns[0]] = 0.

	# columns are renamed
	res.columns = slope_columns

	# resulting dataframe is merged with the original one
	return data.merge(res, left_index=True, right_index=True)


def add_time_differences(data, time_columns, t_diff_columns):

	time_diffs = data.loc[:, time_columns].diff(axis=1)

	# the first measurement is taken as a reference point
	time_diffs[time_columns[0]] = 0.0

	time_diffs.columns = t_diff_columns

	return data.merge(time_diffs, left_index=True, right_index=True)


def mask_missing_data(data):

	return data.fillna(not_a_measurement_value)


def compute_normalization_parameters(df, columns: Sequence):

	normalization_parameters = []

	for c in columns:

		# mean ignoring NaN's
		mean = np.nanmean(df[c].values)

		# variance
		variance = np.nanvar(df[c].values)

		normalization_parameters.append({'mean': mean, 'variance': variance})

	return normalization_parameters


def apply_normalization_parameters(df, columns: Sequence, normalization_parameters: Sequence):

	res = df.copy()

	for c, param in zip(columns, normalization_parameters):

		# normalization (only over "columns")
		res[c] = (res[c] - param['mean']) / np.sqrt(param['variance'])

	return res


def get_biomarkers_paths(parameters, biomarkers, data_path):
	"""
	Returns a list of tuples, each one specifying where to read parameters for a certain biomarker.

	Parameters
	----------
	parameters: dict
		A dictionary containing the required parameters for all the elements in `biomarkers`
	biomarkers: list
		A sequence of strings, one per biomarker, serving a two-fold purpose: choosing the appropriate fields from
		`parameters` and, naming the corresponding biomarker
	data_path: Path object
		The path to (all) the data files

	Returns
	-------
	out: list
		A list with tuples of the form (<path to healthy subjects file>,<path to ill subjects file>,<biomarker name>)

	"""

	biomarkers_settings = []

	for metric in biomarkers:

		# a tuple with
		# i)   the file name for the healthy subjects
		# ii)  the file name for the ill subjects
		# iii) the metric name
		biomarkers_settings.append((
			data_path / parameters[metric]["healthy subjects file"],
			data_path / parameters[metric]["ill subjects file"],
			metric))

	return biomarkers_settings
