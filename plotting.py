import itertools

import numpy as np

from bokeh.plotting import figure, show
import bokeh.palettes

from IPython.display import display


def plot_masked_array(data):

	# for the sake of convenience
	n_lines = len(data)

	# a palette is selected...
	palette_iterator = itertools.cycle(bokeh.palettes.Dark2_5)

	# ...and as many colors as needed extracted from it
	colors = [next(palette_iterator) for _ in range(n_lines)]

	p = figure(width=900, height=800)

	for y, c in zip(data, colors):
		i_valid = np.where(~y.mask)[0]
		p.circle(x=i_valid, y=y.compressed(), color=c, size=8)
		p.line(x=i_valid, y=y.compressed(), color=c)

	# this will show in the browser
	show(p)

	return p


def plot_loss(n_epochs, train, valid=None):

	p = figure(width=900, height=800)

	p.line(x=range(1, n_epochs + 1), y=train, line_color='black')

	if valid:

		p.line(x=range(1, n_epochs + 1), y=valid, line_color='red')

	show(p)


def plot_data(data, t_columns, measurements_columns, show_plot=True):

	colors = itertools.cycle(bokeh.palettes.Dark2_5)

	p = figure(width=900, height=800)

	for (i_row, row), color in zip(data.iterrows(), colors):
		# every example may have a different number of observations
		n = int(row['n_measurements'])

		# the color depends on whether the subject is healthy or ill
		color = 'red' if row['ill'] else 'blue'

		# x and y are extracted from "row" taking into account the specified number of observations
		x = row[t_columns[:n]]
		y = row[measurements_columns[:n]]

		p.line(x, y, color=color)
		p.circle(x, y, color=color)

	if show_plot:
		show(p)

	return p


def highlight_wrong(data, p, t_columns, measurements_columns, color='purple'):

	for (i_row, row) in data.iterrows():
		# every example may have a different number of observations
		n = int(row['n_measurements'])

		# x and y are extracted from "row" taking into account the specified number of observations
		x = row[t_columns[:n]]
		y = row[measurements_columns[:n]]

		p.line(x, y, color=color, line_width=3)
		p.circle(x, y, color=color)

	show(p)


def most_correct_healthy(df, t_columns, measurements_columns, results, decision_threshold, n_most_correct=3):

	healthy = results[~results['actual']]

	# among the healthy subjects, those that were actually predicted as healthy
	correct_within_healthy = healthy[healthy['prediction'] < decision_threshold]

	print('Most Correctly classified as healthy')
	most_correct_within_healthy = correct_within_healthy.sort_values('prediction')[:n_most_correct]
	display(most_correct_within_healthy)
	plot_data(df.loc[most_correct_within_healthy.index], t_columns, measurements_columns)


def most_incorrect_healthy(df, t_columns, measurements_columns, results, decision_threshold, n_most_correct=3):

	healthy = results[~results['actual']]

	# among the healthy subjects, those that were *not* predicted as healthy
	wrong_within_healthy = healthy[healthy['prediction'] > decision_threshold]

	print('Most incorrectly classified as ill')
	most_wrong_within_healthy = wrong_within_healthy.sort_values('prediction', ascending=False)[:n_most_correct]
	display(most_wrong_within_healthy)

	# if the dataframe is not empty
	if len(most_wrong_within_healthy):
		plot_data(df.loc[most_wrong_within_healthy.index], t_columns, measurements_columns)


def most_correct_ill(df, t_columns, measurements_columns, results, decision_threshold, n_most_correct=3):

	ill = results[results['actual']]

	# among the ill subjects, those that were actually predicted as ill
	correct_within_ill = ill[ill['prediction'] > decision_threshold]

	most_correct_within_ill = correct_within_ill.sort_values('prediction', ascending=False)[:n_most_correct]
	print('Most correctly classified as ill')
	display(most_correct_within_ill)
	plot_data(df.loc[most_correct_within_ill.index], t_columns, measurements_columns)


def most_incorrect_ill(df, t_columns, measurements_columns, results, decision_threshold, n_most_correct=3):

	ill = results[results['actual'] == 1]

	# among the ill subjects, those that were *not* predicted as ill
	wrong_within_ill = ill[ill['prediction'] < decision_threshold]

	most_wrong_within_ill = wrong_within_ill.sort_values('prediction')[:n_most_correct]
	print('Most incorrectly classified as healthy')
	display(most_wrong_within_ill)

	# if the dataframe is not empty
	if len(most_wrong_within_ill):
		plot_data(df.loc[most_wrong_within_ill.index], t_columns, measurements_columns)