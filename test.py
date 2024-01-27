#! /usr/bin/env python3

import os
import pathlib
import sys
import argparse
import json
import pickle

import colorama
import yaml

import sklearn.metrics

import models
import data_processing
import data_connector as data_connector_module
import evaluation

# color is reset at the end of every print
colorama.init(autoreset=True)

# ================================= command line arguments

parser = argparse.ArgumentParser(description='RNN testing')

parser.add_argument('-d', '--data-path', default=os.getcwd(), help='path to data files')

command_line_arguments = parser.parse_args(sys.argv[1:])

# data path is extracted and turned into a Pathlib object
data_path = pathlib.Path(command_line_arguments.data_path)

# ------------------ parameters

with open('parameters.yaml') as yaml_data:

	# the parameters file is read to memory
	parameters = yaml.load(yaml_data, Loader=yaml.FullLoader)

# the file containing the previously trained model
model_file = parameters["saved model file"]

# file storing the parameters used for normalization
normalization_parameters_file = parameters["normalization parameters file"]

# file with extra parameters that do not belong in the model
saved_off_model_parameters_file = parameters["saved off-model parameters file"]

# just checking that the model file exists
if not os.path.isfile(model_file):

	raise Exception('file "{}{}{}" does not exist!!'.format(colorama.Fore.RED, model_file, colorama.Style.RESET_ALL))

# the batch size (it doesn't have an impact on the performance when predicting since we are not performing batch
# normalization)
batch_size = parameters["batch size"]

# file in which the whole dataset (as a Dataframe) will be saved
saved_data_file = parameters["test"]["saved data file"]

# predictions will be saved in this file for later reuse/check...
output_predictions_file = parameters["test"]["output predictions file"]

# ...and the derived results in
output_results_file = parameters["test"]["output results file"]

# boolean stating whether to clear measurements after the n-th in every (subject's) sequence
clear_measurements_after = parameters["test"]["clear measurements after"]

# for every subject, the last measurement is picked depending on the diagnosis date
set_last_measurement_on_diagnosis = parameters["test"]["set last measurement according to diagnosis"]

# the last two options shouldn't be set at the same time
assert not (clear_measurements_after and set_last_measurement_on_diagnosis["enabled"])

# the class implementing the model
model_class = getattr(models, parameters["model"])

# ------ data connector

data_connector_class = getattr(data_connector_module, parameters["data connector"]["class"])
data_connector_parameters = parameters["data connector"][parameters["data connector"]["parameters"]]

# ------------------

# trained model is loaded
model = model_class.load_model(model_file, batch_size)

# --

# # it requests the "metric settings" for this script (the name of the latter retrieved using the Pathlib's `stem`
# # method)
# metric_settings = data_processing.get_biomarkers_paths(
# 	parameters[pathlib.Path(__file__).stem], model_class.required_biomarkers, data_path)
#
# data, _ = data_processing.read_and_clean(parameters["data"], metric_settings)
#
# # the maximum number of measurements is inferred from the data
# max_n_measurements = data['n_measurements'].max()

# parameters for test data only
testing_data_parameters = data_connector_parameters[pathlib.Path(__file__).stem]

# common and "specific" (only for this script) parameters are merged together...
data_parameters = {**data_connector_parameters['common'], **testing_data_parameters}

# ...and used to build the `data_connector` instance
data_connector = data_connector_class(**data_parameters)

# the above object is used to build a `DataFrame` with the measurements
data = data_connector.get_data()

# the maximum number of measurements is requested from the `data_connector`
max_n_measurements = data_connector.n_measurements_max

# data is "spread out": every sequence is broken down into several sequences, one including only the 1st measurement,
# another including the 1st and 2nd measurements, and so forth and so on
data = data_connector_module.spread_out_measurements(data, data_connector.time_varying_columns_to_spanned.values())

# this is needed below to match the predictions with the original Excel file
unnormalized_data = data.copy()

# --

# the dictionary containing the names of the columns for every marker is filled
model_class.fill_columns_names(parameters["data"], max_n_measurements)

# data is preprocessed as required by this particular model
data = model_class.preprocess_test(data, normalization_parameters_file)

# data is saved for further use
data.to_pickle(saved_data_file)

# off_model parameters saved during model fitting are loaded
with open(saved_off_model_parameters_file, 'rb') as f:

	off_model_parameters = pickle.load(f)

# -----------

# the missing data are "flagged" so that they are conveniently ignored by Keras
data = data_processing.mask_missing_data(data)

# if the corresponding parameter is set...
if clear_measurements_after:

	# if the received "clear measurements after" is greater than or equal to the actual number of measurements...
	if clear_measurements_after >= data_processing.n_measurements:

		raise Exception(
			'parameter "clear measurements after" (={}) is not consistent: '
			'sequences only have {} measurements'.format(clear_measurements_after, data_processing.n_measurements))

	# FIXME: this must be adapted
	# measurements afterwards are "cleared" (set to a flag value)
	data_processing.clear_measurements_after(clear_measurements_after, data)

# if "set last measurement according to diagnosis" is enabled in the parameters file...
if set_last_measurement_on_diagnosis["enabled"]:

	# FIXME: this must be adapted
	# data is processed and/or filtered according to the given parameters
	data = data_processing.set_last_measurement_according_to_diagnosis(
		data, **set_last_measurement_on_diagnosis["parameters"])


# for every row (label), the ground truth (True/False) and the predicted (soft) value
# predictions = model.predict(data)
predictions = model.predict(data, include_ground_truth=False)

data_and_predictions = unnormalized_data
data_and_predictions['prediction'] = predictions

data_connector.update_excel_file(
	pathlib.Path(testing_data_parameters["file_name"]), data_and_predictions, output_predictions_file)

raise SystemExit

# an empty dictionary to store all the results that later will be saved
results = dict()

# the accuracy is computed
results['0.5-threshold accuracy'] = evaluation.accuracy(predictions, threshold=0.5)

# false positive rate (i.e., 1-specificity), true positive rate (aka, sensitivity/recall), thresholds (automatic)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(predictions['actual'], predictions['prediction'])

# area under the curve
results['roc_auc'] = sklearn.metrics.auc(fpr, tpr)

# results are saved for further use
predictions.to_pickle(output_predictions_file)

print('{}0.5{}-thresholded accuracy is {}{}'.format(
	evaluation.color_threshold,colorama.Style.RESET_ALL, evaluation.color_accuracy, results['0.5-threshold accuracy']))
print('Area Under the Curve is {}{}'.format(evaluation.color_auc, results['roc_auc']))

# ================================= a summary of the results is output to the screen

desired_specificity = parameters["test"]["desired specificity"]

if desired_specificity:

	# the threshold required for achieving (at least) the requested specificity
	threshold = data_processing.fixed_specificity_threshold(predictions, desired_specificity)

	print(
		'for a specificity greather than or equal to {}{}{} (threshold = {}{}{}):'.format(
			evaluation.color_specificity, desired_specificity, colorama.Style.RESET_ALL, evaluation.color_threshold,
			threshold, colorama.Style.RESET_ALL)
	)

else:

	# the average of the thresholds obtained during training for the different folds
	threshold = off_model_parameters['threshold']

	print('for an estimated threshold = {}{}{}):'.format(evaluation.color_threshold,threshold, colorama.Style.RESET_ALL))

# the different metrics are computed using the threshold...
results['accuracy'] = evaluation.accuracy(predictions, threshold)
results['sensitivity'] = evaluation.sensitivity(predictions, threshold)
results['specificity'] = evaluation.specificity(predictions, threshold)

# ...and printed afterwards
evaluation.print_metrics(results['sensitivity'], results['specificity'], results['roc_auc'], results['accuracy'])

with open(output_results_file, 'w') as f:

	json.dump(results, f)
