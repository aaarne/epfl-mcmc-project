import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from argparse import ArgumentParser


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


def plot_energy(data, plot_file):
	'''
	Plot the data showing how the energy decays during the estimation of the
	input vector
	input:	data - data to be plotted
			plot_file - file where to save the plot
	'''
	# Split x and y data by beta
	beta = data[0, 1]
	xs = defaultdict(list)
	ys = defaultdict(list)
	for i in range(data.shape[0]):
		if beta != data[i, 1]:
			xs[data[i, 1]].append(data[i - 1, 0])
			ys[data[i, 1]].append(data[i - 1, 2])
			beta = data[i, 1]
		xs[data[i, 1]].append(data[i, 0])
		ys[data[i, 1]].append(data[i, 2])
	plt.figure()
	for key in xs:
		plt.plot(xs[key], ys[key])
	plt.xlabel('Steps')
	plt.ylabel('Energy')
	plt.legend(xs.keys(), ncol=3)
	# Save plot
	plt.savefig(plot_file)
	plt.show()


if __name__ == '__main__':
	argparser = ArgumentParser()
	argparser.add_argument('--data', type=str, default='output.txt',
		help='The file containing the data to plot and the estimated input')
	argparser.add_argument('--input_ref', type=str, default='input_vect.txt',
		help='The file containing the real input vector')
	argparser.add_argument('--plot_dir', type=str, default='plots',
		help='The directory where to output the plot')

	args = argparser.parse_args()
	# Extract the data to plot and the estimated input vector
	f = open(args.data, 'r')
	estimated_input = np.array([int(x) for x in f.readline().split()])
	plot_data = np.array([[float(y) for y in x.split()] \
		for x in f.read().split('\n')[:-1]])
	f.close()
	# Extract the real input vector
	f = open(args.input_ref, 'r')
	real_input = np.array([int(x) for x in f.readline().split()])

	# Compare the input vectors
	compare_inputs(real_input, estimated_input)

	# Plot the decaying temperature
	plot_file = args.data.split('output')[1].split('.')[0]
	plot_file = f'plot{plot_file}.png'
	plot_file = os.path.join(args.plot_dir, plot_file)
	plot_energy(plot_data, plot_file)
