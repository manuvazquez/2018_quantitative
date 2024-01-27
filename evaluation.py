import abc
import collections
from typing import Sequence

import colorama

import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics

import data_processing

# color is reset at the end of every print
colorama.init(autoreset=True)

# for the sake of convenience
color_sensitivity = colorama.Fore.LIGHTRED_EX
color_specificity = colorama.Fore.LIGHTBLUE_EX
color_auc = colorama.Fore.LIGHTBLACK_EX
color_accuracy = colorama.Fore.LIGHTGREEN_EX
color_threshold = colorama.Fore.LIGHTYELLOW_EX

# interactive plots
plt.ion()

CrossValidationResults = collections.namedtuple(
	'CrossValidationResults', ['accuracy', 'sensitivity', 'specificity', 'auc', 'threshold'])


class ThresholdComputingAlgorithm(metaclass=abc.ABCMeta):

	@abc.abstractmethod
	def compute(self, results) -> float:

		return


class FixedThreshold(ThresholdComputingAlgorithm):

	def __init__(self, threshold: int = 0.5):

		self.threshold = threshold

	def compute(self, results) -> float:

		return self.threshold


class FixedSpecificity(ThresholdComputingAlgorithm):

	def __init__(self, specificity: int = 0.9):

		self.specificity = specificity

	def compute(self, results) -> float:

		return data_processing.fixed_specificity_threshold(results, self.specificity)


def cross_validation(
		model_class, train_df, batch_size: int, sequence_len: int, n_neurons: Sequence[int], n_epochs: Sequence[int],
		dropout: Sequence[float], threshold_calculator, n_splits: int =10, show_roc=False, save_folds=False):

	# a object generating *indexes* to perform cross-validation
	cross_validator = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, random_state=5)

	# the different performance metrics achieved with every set of hyperparameters in every fold
	hyperparameters_accuracy = []
	hyperparameters_sensitivity = []
	hyperparameters_specificity = []
	hyperparameters_auc = []
	hyperparameters_threshold = []

	for i_fold, (train_indexes, valid_indexes) in enumerate(cross_validator.split(train_df, train_df['ill'])):

		# subset of the training data for *actual* training
		fold_train_df = train_df.iloc[train_indexes]

		# subset of the training data for validation
		fold_valid_df = train_df.iloc[valid_indexes]

		# if every "fold" is to be saved...
		if save_folds:

			# training data for this fold is saved, on one hand...
			fold_train_df.to_pickle('fold_{}_train.pickle'.format(i_fold))

			# ...and validation data on the other hand
			fold_valid_df.to_pickle('fold_{}_valid.pickle'.format(i_fold))

		# the different performance metrics obtained for this fold with every set of hyperparameters
		hyperparameters_fold_accuracy = []
		hyperparameters_fold_sensitivity = []
		hyperparameters_fold_specificity = []
		hyperparameters_fold_auc = []
		hyperparameters_fold_threshold = []

		for n, e, d in zip(n_neurons, n_epochs, dropout):

			# a new model is instantiated
			model = model_class(n_neurons=n, batch_size=batch_size, sequence_len=sequence_len, dropout=d)

			# the model is fitted and predictions on the validation set are obtained
			results = model.fit_predict(e, fold_train_df, fold_valid_df)

			# false positive rate (aka, 1-specificity), true positive rate (aka, sensitivity/recall), thresholds
			fpr, tpr, thresholds = sklearn.metrics.roc_curve(results['actual'], results['prediction'])

			roc_auc = sklearn.metrics.auc(fpr, tpr)

			if show_roc:

				# a new figure is created
				plt.figure()

				# ROC curve
				plt.plot(fpr, tpr)
				plt.title('ROC on validation fold #{} (AUC={})'.format(i_fold, roc_auc))

				plt.draw()

				# wait for the figure to be drawn
				plt.pause(0.05)

			# the threshold specified/*computed* by the metric
			threshold = threshold_calculator.compute(results)

			# ALL the performance metrics (regardless of the one implemented in "metric")
			fold_accuracy = accuracy(results, threshold)
			fold_sensitivity = sensitivity(results, threshold)
			fold_specificity = specificity(results, threshold)

			# the different performance metrics for these parameters are stored
			hyperparameters_fold_accuracy.append(fold_accuracy)
			hyperparameters_fold_sensitivity.append(fold_sensitivity)
			hyperparameters_fold_specificity.append(fold_specificity)
			hyperparameters_fold_auc.append(roc_auc)
			hyperparameters_fold_threshold.append(threshold)

			print('{}Fold #{} / {} neurons / {} epochs / dropout = {}:'.format(
				colorama.Fore.LIGHTWHITE_EX,
				i_fold, n, e, d))
			print_metrics(fold_sensitivity, fold_specificity, roc_auc, fold_accuracy)

		# the performance metric obtained for this fold by every set of hyperparameters is kept
		hyperparameters_accuracy.append(hyperparameters_fold_accuracy)
		hyperparameters_sensitivity.append(hyperparameters_fold_sensitivity)
		hyperparameters_specificity.append(hyperparameters_fold_specificity)
		hyperparameters_auc.append(hyperparameters_fold_auc)
		hyperparameters_threshold.append(hyperparameters_fold_threshold)

	return CrossValidationResults(
		accuracy=np.array(hyperparameters_accuracy), sensitivity=np.array(hyperparameters_sensitivity),
		specificity=np.array(hyperparameters_specificity), auc=np.array(hyperparameters_auc),
		threshold=np.array(hyperparameters_threshold))


def sensitivity(results, threshold):

	# only ill subjects
	ill_subjects = results[results['actual']]

	return (ill_subjects['prediction'] >= threshold).mean()


def specificity(results, threshold):

	# only healthy subjects
	healthy_subjects = results[~results['actual']]

	return (healthy_subjects['prediction'] < threshold).mean()


def accuracy(results, threshold):

	return ((results['prediction'] >= threshold) == results['actual']).mean()


def print_metrics(sensitivity, specificity, auc, accuracy, prefix='\t\t'):

	print(prefix + '{}sensitivity{} = {}'.format(color_sensitivity, colorama.Style.RESET_ALL, sensitivity))
	print(prefix + '{}specificity{} = {}'.format(color_specificity, colorama.Style.RESET_ALL, specificity))
	print(prefix + '{}Area Under the Curve{} = {}'.format(color_auc, colorama.Style.RESET_ALL, auc))
	print(prefix + '{}accuracy{} = {}'.format(color_accuracy, colorama.Style.RESET_ALL, accuracy))


def compute_and_print_metrics(results, threshold):

	# false positive rate (i.e., 1-specificity), true positive rate (aka, sensitivity/recall), thresholds (automatic)
	fpr, tpr, _ = sklearn.metrics.roc_curve(results['actual'], results['prediction'])

	print_metrics(
		sensitivity(results, threshold), specificity(results, threshold), sklearn.metrics.auc(fpr, tpr),
		accuracy(results, threshold))