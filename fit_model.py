#! /usr/bin/env python3

import sys
import os
import pathlib
import json
import pickle
import itertools

import argparse
import colorama
import numpy as np
import scipy.io
import yaml

import data_processing
import data_connector
import evaluation
import models

# color is reset at the end of every print
colorama.init(autoreset=True)

# for the sake of reproducibility
np.random.seed(7)

# ================================= command line arguments

parser = argparse.ArgumentParser(description='RNN training')

parser.add_argument('-d', '--data-path', default=os.getcwd(), help='path to data files')

command_line_arguments = parser.parse_args(sys.argv[1:])

# data path is extracted and turned into a pathlib object
data_path = pathlib.Path(command_line_arguments.data_path)

# ================================= parameters

with open('parameters.yaml') as yaml_data:

	# the parameters file is read to memory
	parameters = yaml.load(yaml_data, Loader=yaml.FullLoader)

# size of the batch
batch_size = parameters["batch size"]

# *tentative* model parameters; the value of "n_epochs" specified here is used when trying to find the optimal value
# of "n_neurons"; the latter is then used, in turn, to find the optimal value of "n_epochs"
n_neurons = parameters["fit_model"]["tentative number of neurons"]
n_epochs = parameters["fit_model"]["tentative number of epochs"]
dropout = parameters["fit_model"]["tentative dropout"]

# should the (hyper)parameters be estimated using cross-validation?
estimate_parameters_through_cross_validation = parameters["fit_model"]["cross-validation"]["estimate parameters"]

# if so...
if estimate_parameters_through_cross_validation:

	# ...these are the (hyper)parameter values to be tested
	n_neurons_to_cross_validate = parameters["fit_model"]["cross-validation"]["numbers of neurons to cross validate"]
	n_epochs_to_cross_validate = parameters["fit_model"]["cross-validation"]["numbers of epochs to cross validate"]
	dropout_to_cross_validate = parameters["fit_model"]["cross-validation"]["dropout to cross validate"]

	# should the parameters be estimated jointly (as opposed to independently)?
	joint_parameters_estimation = parameters["fit_model"]["cross-validation"]["joint estimation"]

	# the name of the metric to be minimized
	metric_to_minimize = parameters["fit_model"]["cross-validation"]["metric to be minimized"]

# should an estimate of the performance be computed using cross-validation on the training set with the selected
# hyperparameters?
estimate_performance_through_cross_validation = parameters["fit_model"]["cross-validation"]["estimate performance"]

# whether or not the different dataframe folds will be saved when performing cross-validation to estimate
# the performance
save_folds_during_cross_validation = parameters["fit_model"]["cross-validation"]["save folds"]

# number of splits when performing cross-validation to choose the hyperparameters or estimate the performance
n_splits_inner_cross_validation = parameters["fit_model"]["cross-validation"]["number of splits"]

# ------ output

# file in which the whole dataset (as a Dataframe) will be saved
saved_data_file = parameters["fit_model"]["saved data file"]

# file with extra parameters that do not belong in the model
saved_off_model_parameters_file = parameters["saved off-model parameters file"]

# file that will store the parameters used for normalization
normalization_parameters_file = parameters["normalization parameters file"]

# MATLAB file in which to save both training and test sets, along with the names of the fields
MATLAB_saved_data_file = parameters["fit_model"]["MATLAB saved data file"]

# the hyperparameters picked after training will be saved in a json file
selected_hyperparameters_file = parameters["fit_model"]["selected hyperparameters file"]

model_file = parameters["saved model file"]

# ------ model

# the class implementing the model
model_class = getattr(models, parameters["model"])

# ------ data connector

data_connector_class = getattr(data_connector, parameters["data connector"]["class"])
data_connector_parameters = parameters["data connector"][parameters["data connector"]["parameters"]]

# ================================= data preparation

# # it requests the "metric settings" for this script (the name of the latter retrieved using the Pathlib's `stem` method)
# metric_settings = data_processing.get_biomarkers_paths(
# 	parameters[pathlib.Path(__file__).stem], model_class.required_biomarkers, data_path)
#
# data, _ = data_processing.read_and_clean(parameters["data"], metric_settings)
#
# # the maximum number of measurements is inferred from the data
# max_n_measurements = data['n_measurements'].max()

# common and "specific" (only for this script) parameters are merged together...
data_parameters = {**data_connector_parameters['common'], **data_connector_parameters[pathlib.Path(__file__).stem]}

# ...and used to build the `data_connector` instance
data_connector = data_connector_class(**data_parameters)

# the above object is used to build a `DataFrame` with the measurements
data = data_connector.get_data()

# the maximum number of measurements is requested from the `data_connector`
max_n_measurements = data_connector.n_measurements_max

# breakpoint()

# the dictionary containing the names of the columns for every marker is filled
model_class.fill_columns_names(parameters["data"], max_n_measurements)

# data is preprocessed as required by this particular model
data = model_class.preprocess_training(data, normalization_parameters_file)

# -----------

# the missing data are "flagged" so that they are conveniently ignored by "Keras"
data = data_processing.mask_missing_data(data)

# shuffling to avoid first the healthy subjects, then the ill ones
data = data.sample(frac=1)

# data is saved for further use
data.to_pickle(saved_data_file)

# data is saved in MATLAB format along with the columns names
scipy.io.savemat(MATLAB_saved_data_file, {'data': data.values, 'fields': data.columns.values})

# the maximum number of measurements available in any example
# sequence_max_len = data['n_measurements'].max()
sequence_max_len = max_n_measurements

# -----------

threshold_computing_alg_name = parameters["fit_model"]["threshold computing algorithm"]
threshold_computing_alg_class = getattr(evaluation, threshold_computing_alg_name)
threshold_computing_alg = threshold_computing_alg_class(
	**parameters["available threshold computing algorithms"][threshold_computing_alg_name]["parameters"])

# ================================= cross-validation for choosing the hyperparameters "n_neurons" and "n_epochs"

if estimate_parameters_through_cross_validation:

	if not joint_parameters_estimation:

		# ------------------ number of neurons

		n_neurons_performance_across_folds = evaluation.cross_validation(
			model_class, data, batch_size, sequence_max_len, n_neurons_to_cross_validate,
			[n_epochs] * len(n_neurons_to_cross_validate), [dropout] * len(n_neurons_to_cross_validate),
			threshold_computing_alg, n_splits=n_splits_inner_cross_validation)

		metric = getattr(n_neurons_performance_across_folds, metric_to_minimize)

		# the index of the best set of hyperparameters
		i_best = metric.mean(axis=0).argmax()

		n_neurons = n_neurons_to_cross_validate[i_best]

		# ------------------ number of epochs

		n_epochs_performance_across_folds = evaluation.cross_validation(
			model_class, data, batch_size, sequence_max_len, [n_neurons]*len(n_epochs_to_cross_validate),
			n_epochs_to_cross_validate, [dropout] * len(n_neurons_to_cross_validate), threshold_computing_alg,
			n_splits=n_splits_inner_cross_validation)

		metric = getattr(n_epochs_performance_across_folds, metric_to_minimize)

		# the index of the best set of hyperparameters
		i_best = metric.mean(axis=0).argmax()

		n_epochs = n_epochs_to_cross_validate[i_best]

	else:

		# ------------------ number of neurons AND number of epochs

		n_neurons_n_epochs_dropout = list(zip(*list(
			itertools.product(n_neurons_to_cross_validate, n_epochs_to_cross_validate, dropout_to_cross_validate))))

		n_neurons_n_epochs_performance_across_folds = evaluation.cross_validation(
			model_class, data, batch_size, sequence_max_len, *n_neurons_n_epochs_dropout, threshold_computing_alg,
			n_splits=n_splits_inner_cross_validation)

		metric = getattr(n_neurons_n_epochs_performance_across_folds, metric_to_minimize)

		# the index of the best set of hyperparameters
		i_best = metric.mean(axis=0).argmax()

		n_neurons = n_neurons_n_epochs_dropout[0][i_best]
		n_epochs = n_neurons_n_epochs_dropout[1][i_best]
		dropout = n_neurons_n_epochs_dropout[2][i_best]


# ------------------

crossvalidated_threshold = None

if estimate_performance_through_cross_validation:

	# cross-validation with the selected parameters showing the ROC
	performance_across_folds = evaluation.cross_validation(
		model_class, data, batch_size, sequence_max_len, [n_neurons], [n_epochs], [dropout],  threshold_computing_alg,
		n_splits=n_splits_inner_cross_validation, show_roc=False, save_folds=save_folds_during_cross_validation)

	crossvalidated_threshold = performance_across_folds.threshold.mean()

# ================================= model building

# a model is built (using the values for the parameters found during cross-validation or the initial ones),...
model = model_class(n_neurons=n_neurons, batch_size=batch_size, sequence_len=sequence_max_len, dropout=dropout)

# ...and fitted
results = model.fit_predict(n_epochs, data, data)

threshold = threshold_computing_alg.compute(results)

if crossvalidated_threshold:

	print('threshold during cross-validation: {}; during training: {}'.format(crossvalidated_threshold, threshold))
	print('choosing the latter...')

# the model is saved
model.save_model(model_file)

# the thresholds for the different folds are saved
with open(saved_off_model_parameters_file, 'wb') as f:
	pickle.dump({'threshold': threshold}, f)

# ================================= results
# (shown here, so that they are not scrolled up)

# cross-validation for selecting hyperparameters
if estimate_parameters_through_cross_validation:

	if not joint_parameters_estimation:

		for variable, name in zip(
				[n_neurons_performance_across_folds, n_epochs_performance_across_folds],
				['number of neurons', 'number of epochs']):
			print('[<fold>, <{}>] accuracy\n{}'.format(name, getattr(variable, metric_to_minimize)))
			print('mean across folds\n{}'.format(getattr(variable, metric_to_minimize).mean(axis=0)))

	else:

		print('[<fold>, <number of neurons/epochs/dropout>] accuracy\n{}'.format(
			getattr(n_neurons_n_epochs_performance_across_folds, metric_to_minimize)))
		print('mean across folds\n{}'.format(
			getattr(n_neurons_n_epochs_performance_across_folds, metric_to_minimize).mean(axis=0)))

	print('Selected parameters:\n\t# neurons = {}{}{}\n\t# epochs = {}{}{}\n\tdropout = {}{}'.format(
		colorama.Fore.LIGHTMAGENTA_EX, n_neurons, colorama.Style.RESET_ALL,
		colorama.Fore.LIGHTMAGENTA_EX, n_epochs, colorama.Style.RESET_ALL,
		colorama.Fore.LIGHTMAGENTA_EX, dropout))

# cross-validation for estimating the performance
if estimate_performance_through_cross_validation:

	print('cross-validated results (on training)')

	metrics = [getattr(performance_across_folds, n).mean() for n in [
		'sensitivity', 'specificity', 'auc', 'accuracy']] + ['\t\t']

	evaluation.print_metrics(*metrics)

print('results on training')

evaluation.compute_and_print_metrics(results, threshold)

selected_parameters = {
	"number of neurons": n_neurons, "number of epochs": n_epochs, "dropout": dropout, "threshold": float(threshold)}

with open(selected_hyperparameters_file, 'w') as f:

	json.dump(selected_parameters, f)
