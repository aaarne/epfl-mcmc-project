import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from argparse import ArgumentParser


def plot_file(data_file, plot_dir, key):
	'''
	Generate the name of the file to include the plot in
	input:	key - keyword for the plot
	output:	the name of the plot file
	'''
	f = data_file.split('output')[1].split('.')[0]
	f = f'plot{f}_{key}.png'
	f = os.path.join(plot_dir, f)
	return f


def compare_inputs(real, estimate):
	'''
	Print statistics regarding the estimated input vector with respect to the
	real one
	input:	real - real input vector
			estimate - estimated input vector
	'''
	# Compute the confusion matrix
	confusion_matrix = np.zeros([2, 2])
	for i in range(real.shape[0]):
		if real[i] == estimate[i] == 1:
			confusion_matrix[0, 0] += 1
		elif real[i] == estimate[i] == -1:
			confusion_matrix[1, 1] += 1
		elif real[i] == -1:
			confusion_matrix[0, 1] += 1
		elif real[i] == 1:
			confusion_matrix[1, 0] += 1
	tp, fp, fn, tn = confusion_matrix.reshape(-1)

	# Compute metrics
	accuracy = (tp + tn) / (tp + fp + fn + tn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * precision * recall / (precision + recall)

	print('Comparing the real and the estimated input vectors we get the following metrics:')
	print(f'Confusion matrix is:\n {confusion_matrix}')
	print(f'Accuracy: {accuracy}')
	print(f'Precision: {precision}')
	print(f'Recall: {recall}')
	print(f'F1-score: {f1_score}')


def plot_evolution(data_file, input_file, plot_file, index=2, label='Energy'):
	'''
	Plot the data showing how the energy decays during the estimation of the
	input vector
	input:	data_file - file containing the data to be plotted
			input_file - file containing the real input vector
			plot_file - file where to save the plot
			index - index of the data element to be plotted
			label - label associated with the plotted element
	'''
	# Extract the data to plot and the estimated input vector
	f = open(data_file, 'r')
	estimated_input = np.array([int(x) for x in f.readline().split()])
	plot_data = np.array([[float(y) for y in x.split()] \
		for x in f.read().split('\n')[:-1]])
	f.close()
	# Extract the real input vector
	f = open(input_file, 'r')
	real_input = np.array([int(x) for x in f.readline().split()])

	# Compare the input vectors
	compare_inputs(real_input, estimated_input)

	# Split x and y data by beta
	beta = plot_data[0, 1]
	xs = defaultdict(list)
	ys = defaultdict(list)
	for i in range(plot_data.shape[0]):
		if beta != plot_data[i, 1]:
			xs[plot_data[i, 1]].append(plot_data[i - 1, 0])
			ys[plot_data[i, 1]].append(plot_data[i - 1, index])
			beta = plot_data[i, 1]
		xs[plot_data[i, 1]].append(plot_data[i, 0])
		ys[plot_data[i, 1]].append(plot_data[i, index])
	plt.figure()
	for key in xs:
		plt.plot(xs[key], ys[key])
	plt.xlabel('Steps')
	plt.ylabel(label)
	plt.legend([f'{l:.2f}' for l in xs.keys()], ncol=3)
	# Save plot
	plt.savefig(plot_file)


def plot_alpha(data_file, plot_dir, type):
	'''
	Plot the dependency of the reconstruction error on alpha
	input:	data_file - file containing the data to be plotted
			plot_dir - directory where to save the plot
			type - whether to plot last reconstruction error ('last'),
			minimum reconstruction error ('min') or both
	'''
	data = pd.read_csv(data_file)
	x = data['alpha']
	if type == 'last':
		y = data['mean_err']
		std = data['std_err']
	elif type == 'min':
		y = data['mean_min_err']
		std = data['std_min_err']
	else:
		y = data['mean_err']
		std = data['std_err']
		y2 = data['mean_min_err']
		std2 = data['std_min_err']

	# Plot the data
	plt.errorbar(x, y, std)
	if type == 'both':
		plt.errorbar(x, y2, std2)

	plt.xlabel('Alpha')
	plt.ylabel('Reconstruction Error')
	# Save plot
	plot_file = os.path.join(plot_dir, f'alpha_{type}.png')
	plt.savefig(plot_file)


if __name__ == '__main__':
	argparser = ArgumentParser()
	argparser.add_argument('--data', type=str, default='output.txt',
		help='the file containing the data to plot and the estimated input')
	argparser.add_argument('--input_ref', type=str, default='input_vect.txt',
		help='the file containing the real input vector')
	argparser.add_argument('--plot_dir', type=str, default='plots',
		help='the directory where to output the plot')
	argparser.add_argument('--plot_alpha', type=str, default='no_alpha',
		help='whether to plot reconstruction error versus alpha')
	args = argparser.parse_args()

	if args.plot_alpha == 'no_alpha':
		# Plot the decaying temperature
		plot_evolution(args.data, args.input_ref, plot_file(args.data, args.plot_dir, 'energy'), index=2, label='Energy')
		plot_evolution(args.data, args.input_ref, plot_file(args.data, args.plot_dir, 'error'), index=3, label='Reconstruction Error')
		plt.show()
	else:
		# Plot the reconstruction error as a function of alpha
		plot_alpha(args.data, args.plot_dir, args.plot_alpha)
		plt.show()
