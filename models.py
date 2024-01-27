import pickle

import numpy as np
import pandas as pd

import keras.models
from keras.models import Sequential
from keras.layers import Dense, Masking, Dropout
from keras.layers import SimpleRNN

from keras.models import Model
from keras.layers import Input, concatenate

import data_processing


class SingleBiomarker:

	required_biomarkers = None

	columns = {}

	# are the classes (labels) to be weighted in the cost function according to their prevalence
	weight_classes = False

	# which layer to check in order to infer the number of neurons when loading the model
	i_layer_to_infer_n_neurons = 1

	@classmethod
	def load_model(cls, filename: str, batch_size: int):
		"""
		Creates an instance of the class from a saved Keras model.

		Parameters
		----------
		filename: str
			The file containing the Keras model.
		batch_size: int
			The batch size.

		Returns
		-------
		out: SingleBiomarker object
			A model instance (that wraps the Keras model).

		"""

		model = keras.models.load_model(filename)

		# the number of neurons can be easily inferred from the appropriate layer...
		n_neurons = model.layers[cls.i_layer_to_infer_n_neurons].get_weights()[0].shape[1]

		# ...and so can the length of the input sequences
		sequence_len = model.layers[cls.i_layer_to_infer_n_neurons].input_shape[1]

		new_instance = cls(n_neurons=n_neurons, batch_size=batch_size, sequence_len=sequence_len, build_model=False)
		new_instance.keras_model = model

		return new_instance

	@classmethod
	def fill_columns_names(cls, data_parameters, n_measurements):
		"""

		Fills in the class variable `columns` (a dictionary)

		Parameters
		----------
		data_parameters: dict
			A dictionary containing the *data* settings as specified in the parameters file.
		n_measurements: int
			The maximum number of measurements across all the samples (subjects).

		"""

		for prefix in cls.required_biomarkers:

			cls.columns[f'{prefix}'] = [
				f'{data_parameters[prefix]["columns prefix"]}{i}' for i in range(1, 1 + n_measurements)]

		cls.columns['t'] = [f't_{i}' for i in range(1, 1 + n_measurements)]

	@classmethod
	def df_to_array(cls, df):
		"""
		It turns a Pandas dataframe into numpy array in the format required by Keras.

		Parameters
		----------
		df : Pandas Dataframe
			A dataframe containing (training or test) data.

		Returns
		-------
		out : Numpy array

		"""

		# there should be a single required metric
		assert len(cls.required_biomarkers) == 1

		# the *only* required biomarker is used as key in the `cls.columns` dictionary, and those are the columns (or
		# rather, their `values`) extracted from the given dataframe. Notice that a new dimension must be added before
		# passing the dataset to Keras: the number of features, which here is 1.
		return np.expand_dims(df[cls.columns[cls.required_biomarkers[0]]].values, 2)

	@classmethod
	def preprocess_training(cls, data, parameters_file):

		# the normalization parameters are computed,...
		normalization_parameters = data_processing.compute_normalization_parameters(
			data, [cls.columns[cls.required_biomarkers[0]]])

		# ...saved,...
		with open(parameters_file, 'wb') as f:
			pickle.dump(normalization_parameters, f)

		# ...and applied on the data
		return data_processing.apply_normalization_parameters(
			data, [cls.columns[cls.required_biomarkers[0]]], normalization_parameters)

	@classmethod
	def preprocess_test(cls, data, parameters_file):

		# normalization parameters are loaded (they should have been obtained during the fitting of the model)...
		with open(parameters_file, 'rb') as f:

			normalization_parameters = pickle.load(f)

		# ...and applied on the data
		return data_processing.apply_normalization_parameters(
			data, [cls.columns[cls.required_biomarkers[0]]], normalization_parameters)

	def __init__(
			self, n_neurons: int, batch_size: int, sequence_len: int, dropout=0.6, build_model: bool = True, n_features=1,
			compile=True):

		self.n_neurons = n_neurons
		self.batch_size = batch_size
		self.sequence_len = sequence_len
		self.n_features = n_features

		self.dropout = dropout

		self.shift = (-1, 1)
		self.rotation = (-1, 1)

		if build_model:

			self.keras_model = Sequential()

			self.keras_model.add(
				Masking(mask_value=data_processing.not_a_measurement_value, input_shape=(sequence_len, n_features)))

			self.keras_model.add(SimpleRNN(
				n_neurons, recurrent_initializer='identity', batch_input_shape=(batch_size, sequence_len, n_features)))

			self.keras_model.add(Dropout(self.dropout))

			self.keras_model.add(Dense(1, activation='sigmoid'))

			if compile:
				self.keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	def generator(self, df):

		raise Exception('not implemented!!')

	def fit_predict(self, n_epochs: int, train_df, valid_df):
		"""
		It fits a dataframe on the training set and evaluates the resulting model on the validation set.

		Parameters
		----------
		n_epochs : int
		train_df : Pandas dataframe
					The training set.
		valid_df : Pandas dataframe
					The validation set.

		Returns
		-------
		results: dataframe
				Results on the validation dataframe

		"""

		self.fit(n_epochs, train_df)

		return self.predict(valid_df)

	def fit_predict_generator(self, n_epochs: int, train_df, valid_df):

		gen = self.generator(train_df)

		self.keras_model.fit_generator(generator=gen.flow(), steps_per_epoch=gen.n_batches_per_epoch, epochs=n_epochs)

		return self.predict(valid_df)

	def validating_fit_predict_generator(self, n_epochs: int, train_df, valid_df):

		train_gen = self.generator(train_df)
		valid_gen = self.generator(valid_df)

		self.keras_model.fit_generator(
			generator=train_gen.flow(), steps_per_epoch=train_gen.n_batches_per_epoch, epochs=n_epochs,
			validation_data=valid_gen.flow(), validation_steps=valid_gen.n_batches_per_epoch)

		return self.predict(valid_df)

	def fit(self, n_epochs: int, train_df, valid_df=None):
		"""
		It fits a dataframe on the training set.

		Parameters
		----------
		n_epochs : int
		train_df : Pandas dataframe
					The training set.
		valid_df: Pandas dataframe
					The validation set.
		class_weight: dict
					weights for the different classes (in Keras format)

		"""

		# the dataframe is turned into a Numpy array in the format expected by Keras
		training = self.df_to_array(train_df)

		# if classes are to be weighted according to their prevalence...
		if self.__class__.weight_classes:

			# NOTE: this assumes `value_counts` returns the values in order
			class_weight = dict(zip([0, 1], 1. / train_df['ill'].value_counts().values))

			print(f'class weights: {class_weight}')

		else:

			class_weight = None

		if valid_df is not None:

			valid = self.df_to_array(valid_df)

			return self.keras_model.fit(
				training, train_df['ill'].values, epochs=n_epochs, batch_size=self.batch_size,
				validation_data=(valid, valid_df['ill']), class_weight=class_weight)

		else:

			# `fit` returns a "History" object
			return self.keras_model.fit(
				training, train_df['ill'].values, epochs=n_epochs, batch_size=self.batch_size, class_weight=class_weight)

	def predict(self, valid_df: pd.DataFrame, include_ground_truth: bool = True):
		"""
		It fits evaluates the current model on the validation set.

		Parameters
		----------
		valid_df : Pandas dataframe
					The validation set.
		include_ground_truth: bool
					Whether to include in the output the actual labels

		Returns
		-------
		results: dataframe
				Results on the validation dataframe

		"""

		# the dataframe is turned into a Numpy array in the format expected by Keras
		validation = self.df_to_array(valid_df)

		predictions = self.keras_model.predict(validation, batch_size=self.batch_size)

		# a dataframe compiling with the predictions and...
		results = pd.DataFrame({'prediction': predictions[:, 0]}, index=valid_df.index)

		# ..., if requested,...
		if include_ground_truth:

			# ...the actual values
			results['actual'] = valid_df['ill']

		return results

	def save_model(self, filename):

		self.keras_model.save(filename)


class TimedSingleBiomarker(SingleBiomarker):

	def __init__(
			self, n_neurons: int, batch_size: int, sequence_len: int, dropout=0.6, build_model: bool = True, compile=True):

		super().__init__(n_neurons, batch_size, sequence_len, dropout, build_model, 2, compile)

	@classmethod
	def df_to_array(cls, df):

		# there should be a single required metric
		assert len(cls.required_biomarkers) == 1

		measurements = df[cls.columns[cls.required_biomarkers[0]]]
		time = df[cls.columns['t']]

		return np.stack([measurements.values, time.values], axis=2)

	@classmethod
	def preprocess_training(cls, data, parameters_file):

		# the normalization parameters are computed,...
		normalization_parameters = data_processing.compute_normalization_parameters(
			data, [cls.columns[cls.required_biomarkers[0]], cls.columns['t']])

		# ...saved,...
		with open(parameters_file, 'wb') as f:
			pickle.dump(normalization_parameters, f)

		# ...and applied on the data
		return data_processing.apply_normalization_parameters(
			data, [cls.columns[cls.required_biomarkers[0]], cls.columns['t']], normalization_parameters)

	@classmethod
	def preprocess_test(cls, data, parameters_file):

		# normalization parameters are loaded (they should have been obtained during the fitting of the model)...
		with open(parameters_file, 'rb') as f:

			normalization_parameters = pickle.load(f)

		# ...and applied on the data
		return data_processing.apply_normalization_parameters(
			data, [cls.columns[cls.required_biomarkers[0]], cls.columns['t']], normalization_parameters)


class TimedCA125(TimedSingleBiomarker):

	required_biomarkers = ['CA125']


class TimedHE4(TimedSingleBiomarker):

	required_biomarkers = ['HE4']


class WeightedClassesTimedCA125(TimedSingleBiomarker):

	weight_classes = True

	required_biomarkers = ['CA125']


class WeightedClassesTimedHE4(TimedSingleBiomarker):

	weight_classes = True

	required_biomarkers = ['HE4']


class MultipleJointBiomarkers(SingleBiomarker):

	required_biomarkers = ['CA125', 'HE4', 'Gly']

	def __init__(
			self, n_neurons: int, batch_size: int, sequence_len: int, dropout=0.6, build_model: bool = True,
			n_features: int = 3, compile=True):

		super().__init__(n_neurons, batch_size, sequence_len, dropout, build_model, n_features=n_features, compile=compile)

	@classmethod
	def preprocess_training(cls, data, parameters_file):

		# the *names* of the columns that enter the normalization
		columns_to_normalize = [cls.columns[m] for m in ['t'] + cls.required_biomarkers]

		# the normalization parameters are computed...
		normalization_parameters = data_processing.compute_normalization_parameters(data, columns_to_normalize)

		# ...saved...
		with open(parameters_file, 'wb') as f:
			pickle.dump(normalization_parameters, f)

		# ...and applied on the data
		return data_processing.apply_normalization_parameters(data, columns_to_normalize, normalization_parameters)

	@classmethod
	def preprocess_test(cls, data, parameters_file):

		# the *names* of the columns that enter the normalization
		columns_to_normalize = [cls.columns[m] for m in ['t'] + cls.required_biomarkers]

		# normalization parameters are loaded (they should have been obtained during the fitting of the model)...
		with open(parameters_file, 'rb') as f:

			normalization_parameters = pickle.load(f)

		# ...and applied on the data
		return data_processing.apply_normalization_parameters(data, columns_to_normalize, normalization_parameters)

	@classmethod
	def df_to_array(cls, df):

		return np.stack([df[cls.columns[m]].values for m in cls.required_biomarkers], axis=2)


class MultipleIndependentBiomarkers(MultipleJointBiomarkers):

	i_layer_to_infer_n_neurons = 6

	def __init__(
			self, n_neurons: int, batch_size: int, sequence_len: int, dropout=0.6, build_model: bool = True, n_features=1,
			compile=True):

		super().__init__(n_neurons, batch_size, sequence_len, dropout, build_model=False, n_features=n_features, compile=compile)

		n_metrics = len(self.__class__.required_biomarkers)

		inputs = [None] * n_metrics
		outputs = [None] * n_metrics

		for i_metric in range(n_metrics):

			inputs[i_metric] = Input(shape=(sequence_len, n_features), dtype='float32')
			outputs[i_metric] = Masking(mask_value=data_processing.not_a_measurement_value)(inputs[i_metric])
			outputs[i_metric] = SimpleRNN(n_neurons, batch_input_shape=(batch_size, sequence_len, n_features))(
				outputs[i_metric])

		out = concatenate(outputs)
		out = Dropout(self.dropout)(out)
		out = Dense(1, activation='sigmoid')(out)

		self.keras_model = Model(inputs=inputs, outputs=out)

		if compile:
			self.keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	@classmethod
	def df_to_array(cls, df):

		return [np.expand_dims(df[cls.columns[m]].values, 2) for m in cls.required_biomarkers]


class CA125HE4Independent(MultipleIndependentBiomarkers):

	required_biomarkers = ['CA125', 'HE4']


class CA125GlyIndependent(MultipleIndependentBiomarkers):

	required_biomarkers = ['CA125', 'Gly']


class TimedMultipleIndependentBiomarkers(MultipleIndependentBiomarkers):

	def __init__(self, n_neurons: int, batch_size: int, sequence_len: int, dropout=0.6, build_model: bool = True, compile=True):

		super().__init__(n_neurons, batch_size, sequence_len, dropout, build_model, n_features=2, compile=compile)

	@classmethod
	def df_to_array(cls, df):

		time = df[cls.columns['t']]

		return [np.stack([df[cls.columns[m]].values, time.values], axis=2) for m in cls.required_biomarkers]


class TimedCA125HE4Independent(TimedMultipleIndependentBiomarkers):

	i_layer_to_infer_n_neurons = 4

	required_biomarkers = ['CA125', 'HE4']


class WeightedClassesTimedCA125HE4Independent(TimedMultipleIndependentBiomarkers):

	weight_classes = True

	required_biomarkers = ['CA125', 'HE4']
