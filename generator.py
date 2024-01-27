import numpy as np
import pandas as pd
import sklearn.utils

import data_processing
import models


class BatchGenerator:

	def __init__(
			self, df: pd.DataFrame, batch_size: int, model_class: models.SingleBiomarker, ill_percentage: float = 0.5):

		"""

		Creates a batch generator, in which every batch is sampled. Hence, duplicates will happen, and there are no
		"epochs".

		Parameters
		----------
		df: dataframe
			The data to partition into batches.
		batch_size: int
			The size of each batch.
		model_class: class
			The class along with which this generator will be used.
		ill_percentage: float
			The percentage of "ill" subjects in every batch.
		"""

		self.batch_size = batch_size
		self.model_class = model_class
		self.ill_percentage = ill_percentage

		# for the sake of clarity and convenience
		self.ill_subjects = df[df['ill']]
		self.healthy_subjects = df[~df['ill']]

		self.n_ill = len(self.ill_subjects)
		self.n_healthy = len(self.healthy_subjects)

		# how many samples of each class in every batch
		# NOTE: for ill subjects, we need to account for the case in which the requested `ill_percentage` yields more
		# ill subjects than there actually is
		self.n_ill_per_batch = min(int(batch_size * ill_percentage), self.n_ill)
		self.n_healthy_per_batch = batch_size - self.n_ill_per_batch

	@property
	def n_batches_per_epoch(self):
		"""
		Returns the number of batches per epoch.

		"""

		# since it is assumed there are many more healthy subjects than ill ones, the number of the former is used to
		# decide whether an epoch is complete or not
		return int(np.ceil(self.n_healthy/self.n_healthy_per_batch))

	def flow(self) -> np.ndarray:

		"""

		Returns the next *sampled* batch.

		Returns
		-------
		out: ndarray
			An array with the structure required by Keras.

		"""

		while True:

			# sampled ill and healthy subjects are put together in a single dataframe...
			df = pd.concat((
				self.ill_subjects.sample(n=self.n_ill_per_batch),
				self.healthy_subjects.sample(n=self.n_healthy_per_batch)))

			# ...which is afterwards shuffled
			df = df.sample(frac=1)

			yield self.model_class.df_to_array(df), df['ill']


class ThoroughBatchGenerator(BatchGenerator):

	def __init__(self, df, batch_size, model_class, ill_percentage=0.5, shift=(-0.75, 0.75), rotation=(-3, 3)):

		"""

		Creates a batch generator that will exhaust the

		Parameters
		----------
		df: dataframe
			The data to partition into batches.
		batch_size: int
			The size of each batch.
		model_class: class
			The class along with which this generator will be used.
		ill_percentage: float
			The percentage of "ill" subjects in every batch.
		shift: tuple
			The range in which the amount of shift to be applied is uniformly sampled.
		rotation: tuple:
			The range in which the amount of rotation to be applied is uniformly sampled.
		"""

		super().__init__(df, batch_size, model_class, ill_percentage)

		# pseudo-random numbers generator
		self._prng = np.random.RandomState(10)

		self.shift = shift
		self.rotation = rotation

	def transform_batch(self, batch: pd.DataFrame):

		"""

		Transforms a dataframe according to the shift and rotation specified.

		Parameters
		----------
		batch: dataframe
			The dataframe containing the data to be transformed.

		Returns
		-------
		out: dataframe
			A dataframe with the data transformed.

		"""

		if self.shift is not None:

			# shift
			batch = batch.apply(
				data_processing.shift, axis=1, measurements_columns=self.model_class.columns['CA125'],
				value=self._prng.uniform(*self.shift))

			# print('shifting')

		if self.rotation is not None:

			# rotation
			batch = batch.apply(
				data_processing.rotate, axis=1, t_columns=self.model_class.columns['t'],
				measurements_columns=self.model_class.columns['CA125'], degrees=self._prng.uniform(*self.rotation))

			# print('rotating')

		return batch

	def flow(self):

		while True:

			# shuffling, so that different epochs yield different batches (samples are grouped differently)
			ill_subjects = sklearn.utils.shuffle(self.ill_subjects)
			healthy_subjects = sklearn.utils.shuffle(self.healthy_subjects)

			i_ill = np.arange(self.n_ill_per_batch)
			i_healthy = np.arange(self.n_healthy_per_batch)

			for i_step in range(self.n_batches_per_epoch):

				df = pd.concat((ill_subjects.iloc[i_ill], healthy_subjects.iloc[i_healthy]))

				# another shuffle so that, within the batch, ill subjects don't go first and healthy subjects afterwards
				df = sklearn.utils.shuffle(df)

				# batch is transformed
				df = self.transform_batch(df)

				yield self.model_class.df_to_array(df), df['ill']

				i_ill = np.mod(i_ill + self.n_ill_per_batch, self.n_ill)
				i_healthy = np.mod(i_healthy + self.n_healthy_per_batch, self.n_healthy)


class ThoroughBatchGeneratorCA125HE4Gly(ThoroughBatchGenerator):

	def transform_batch(self, batch):

		if self.shift is not None:

			# shift
			batch = batch.apply(
				data_processing.shift, axis=1, measurements_columns=data_processing.measurements_columns,
				value=self._prng.uniform(*self.shift))

			batch = batch.apply(
				data_processing.shift, axis=1, measurements_columns=data_processing.HE4_columns,
				value=self._prng.uniform(*self.shift))

			batch = batch.apply(
				data_processing.shift, axis=1, measurements_columns=data_processing.Gly_columns,
				value=self._prng.uniform(*self.shift))

		if self.rotation is not None:

			# rotation
			batch = batch.apply(
				data_processing.rotate, axis=1, measurements_columns=data_processing.measurements_columns,
				degrees=self._prng.uniform(*self.rotation))

			batch = batch.apply(
				data_processing.rotate, axis=1, measurements_columns=data_processing.HE4_columns,
				degrees=self._prng.uniform(*self.rotation))

			batch = batch.apply(
				data_processing.rotate, axis=1, measurements_columns=data_processing.Gly_columns,
				degrees=self._prng.uniform(*self.rotation))

		return batch


class CNNBatchGenerator:

	def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int, ill_percentage: float = 0.5):

		self.data = data
		self.labels = labels
		self.batch_size = batch_size
		self.ill_percentage = ill_percentage

		# the indexes of the *ill* and *healthy* subjects
		self.i_ill_subjects = np.where(labels)[0]
		self.i_healthy_subjects = np.where(~labels)[0]

		# the number of *ill* and *healthy* subjects
		self.n_ill = len(self.i_ill_subjects)
		self.n_healthy = len(self.i_healthy_subjects)

		# how many samples of each class in every batch
		# NOTE: for ill subjects, we need to account for the case in which the requested
		# `ill_percentage` yields more ill subjects than there actually is
		self.n_ill_per_batch = min(int(batch_size * ill_percentage), self.n_ill)
		self.n_healthy_per_batch = batch_size - self.n_ill_per_batch

	@property
	def n_batches_per_epoch(self):
		"""
		Returns the number of batches per epoch.

		"""

		# since it is assumed there are many more healthy subjects than ill ones, the number of the former is used to
		# decide whether an epoch is complete or not
		return int(np.ceil(self.n_healthy/self.n_healthy_per_batch))

	def flow(self):

		while True:

			# shuffling, so that different epochs yield different batches (samples are grouped differently); notice
			# that both variables just contain indexes for rows within `self.data`
			shuffled_i_ill_subjects  = sklearn.utils.shuffle(self.i_ill_subjects)
			shuffled_i_healthy_subjects = sklearn.utils.shuffle(self.i_healthy_subjects)

			# selected indexes among the *ill* and *healthy* subjects
			i_ill = np.arange(self.n_ill_per_batch)
			i_healthy = np.arange(self.n_healthy_per_batch)

			for i_step in range(self.n_batches_per_epoch):

				# indexes for the *ill* and *healthy* subjects are concatenated and shuffled (ill subjects don't go
				# first and healthy subjects afterwards)
				i_batch = sklearn.utils.shuffle(np.hstack(
					(shuffled_i_ill_subjects[i_ill], shuffled_i_healthy_subjects[i_healthy])))

				yield np.expand_dims(self.data[i_batch], 2), self.labels[i_batch]

				i_ill = np.mod(i_ill + self.n_ill_per_batch, self.n_ill)
				i_healthy = np.mod(i_healthy + self.n_healthy_per_batch, self.n_healthy)