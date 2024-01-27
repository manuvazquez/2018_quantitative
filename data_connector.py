from typing import Optional
import pathlib
import abc

import numpy as np
import pandas as pd


class Base(metaclass=abc.ABCMeta):

	@abc.abstractmethod
	def get_data(self) -> pd.DataFrame:

		pass


class ExcelSpreadsheet(Base):

	def __init__(
			self, file_name: str, time_varying_data: dict, time_varying_columns_mapping: dict,
			subject_id_column: str, time_column: str, maximum_number_of_measurements: int,
			ill_column: str = None) -> None:

		super().__init__()

		self.file = pathlib.Path(file_name)
		self.time_varying_data = time_varying_data
		self.time_varying_columns_mapping = time_varying_columns_mapping
		self.subject_id_column = subject_id_column
		self.time_column = time_column
		self.n_measurements_max: int = maximum_number_of_measurements
		self.ill_column = ill_column

		# the "time column" should be among the time-varying ones
		assert time_column in time_varying_columns_mapping

		# for every time-varying column, the names of the columns that will span in the output `DataFrame`
		self.time_varying_columns_to_spanned: dict = {}
		for k, v in self.time_varying_columns_mapping.items():

			self.time_varying_columns_to_spanned[k] = [
				self.time_varying_data[v]["columns prefix"] + str(n) for n in range(1, self.n_measurements_max + 1)]

	def get_data(self) -> pd.DataFrame:

		# data is read from Excel spreadsheet
		df = pd.read_excel(self.file)

		# data is grouped by subject
		grouped_by_subject_df = df.groupby(self.subject_id_column)

		# all the columns in a single list to be used below
		columns = sum(self.time_varying_columns_to_spanned.values(), [])

		# a new *empty* `DataFrame` is built with the above columns, and using the id's of the subjects as "index"
		output_df = pd.DataFrame(index=df[self.subject_id_column].unique(), columns=columns, dtype=float)

		# extra columns
		output_df['n_measurements'] = 0

		# if an "ill column" was passed...
		if self.ill_column:

			output_df['ill'] = False

		# for every subject...
		for subject, group in grouped_by_subject_df:

			# the rows for this subject are ordered by time
			sorted_group = group.sort_values(self.time_column)

			# the number of measurements for this subject
			n_measurements = len(group)

			# for every time-varying column (in the Excel file) along with the ones it spans in the output `DataFrame`..
			for k, v in self.time_varying_columns_to_spanned.items():

				# ...the output `DataFrame` is filled in (horizontally) with the values from the corresponding column in
				# the Excel spreadsheet
				output_df.loc[subject, v[:n_measurements]] = sorted_group[k].tolist()

			# the number of measurements is also added
			output_df.loc[subject, 'n_measurements'] = n_measurements

			# if an "ill column" was passed...
			if self.ill_column:

				# all the rows for this subject should have the same label (ill/healthy)
				assert len(group[self.ill_column].unique()) == 1

				# the ill/healthy label for this subject is set
				output_df.loc[subject, 'ill'] = bool(group[self.ill_column].iat[0])

		return output_df

	def update_excel_file(
			self, file: pathlib.Path, data_and_predictions: pd.DataFrame, output_file_name: str,
			probability_column_name: str = 'probability'):

		# a mapping from the program internal names to those in the Excel file
		reverse_time_varying_columns_mapping = dict(
			zip(self.time_varying_columns_mapping.values(), self.time_varying_columns_mapping.keys()))

		# the columns associated with time in the `DataFrame`
		time_columns = self.time_varying_columns_to_spanned[reverse_time_varying_columns_mapping['Age']]

		# an auxiliar function to be used in Pandas' `apply` below
		def keep_last_time_only(row, time_columns):

			# the value of last "time column" is recorded in a new column `last_t`
			row['last_t'] = row[time_columns[int(row['n_measurements'])-1]]

			# old time columns are dropped on return
			return row.drop(time_columns)

		# for every row, only the time of the last measurement is kept
		data_and_predictions = data_and_predictions.apply(keep_last_time_only, axis=1, time_columns=time_columns)

		# Excel file to be filled is read into a Pandas `DataFrame`
		output_df = pd.read_excel(file)

		# the new column is added
		output_df[probability_column_name] = np.nan

		# for every subject "id" and "age"...
		for i, (subject_id, subject_age) in output_df[
			[self.subject_id_column, reverse_time_varying_columns_mapping['Age']]].iterrows():

			# the data for this subject is located in the input `DataFrame`...
			subject_df = data_and_predictions.loc[subject_id]

			# ...and within it, the row whose `last_t` value matches the above subject "age"; the corresponding
			# "prediction" value is extracted...
			prediction = subject_df[subject_df['last_t'] == subject_age]['prediction'].item()

			# ...and filled in the appropriate position of the output `DataFrame`
			output_df.loc[i, probability_column_name] = prediction

		output_df.to_excel(output_file_name, index=False)


def spread_out_measurements(data: pd.DataFrame, variables: list):

	# the maximum number of measurements in any subject
	max_n_measurements = data['n_measurements'].max()

	# a new `DataFrame` is built by duplicating the data as many times as the maximum number of measurements; a new
	# index level (associated with the number of measurements used to make the prediction) is added
	out = pd.concat([data]*max_n_measurements, keys=range(1, 1 + max_n_measurements)).swaplevel().sort_index()

	# the new index is also added as a column
	out['measurements_up_to'] = out.index.get_level_values(1)

	# the function to be passed to Pandas' `apply`
	def remove_forward(row, time_varying_data_columns: list):

		# the number of measurements to be kept
		measurements_up_to = int(row['measurements_up_to'])

		# for every list of columns (a time-varying field)...
		for v in time_varying_data_columns:

			# values associated with measurements after the number of those to be kept are set to `NaN`
			row[v[measurements_up_to:]] = np.nan

		# the number of measurements of the new row is adjusted accordingly
		row['n_measurements'] = measurements_up_to

		return row

	# processing
	out = out.apply(remove_forward, axis=1, time_varying_data_columns=variables)

	# column `n_measurements` is cast back as integer
	out['n_measurements'] = out['n_measurements'].astype(int)

	# the indexes of rows that are not duplicates after *when ignoring* columns `n_measurements` and `measurements_up_to`
	index_unique = out.drop(['n_measurements', 'measurements_up_to'], axis=1).drop_duplicates().index

	# output `DataFrame` is returned after dropping (auxiliar) column `measurements_up_to`
	return out.loc[index_unique].drop('measurements_up_to', axis=1)
