import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import OrderedDict
import matplotlib.ticker


keys = {
	'simple': 'b', 
	'glauber': 'r',
}

fix, ax = plt.subplots()

for key, color in keys.items():
	for filename in glob.glob(f'comparison_methods/output_{key}_*.txt'):
		with open(filename, 'r') as f:
			f.readline()
			data = np.array([[float(y) for y in x.split()] \
				for x in f.read().split('\n')[:-1]])

			ax.plot(
				data[:,0],
				data[:,2],
				color,
				label={'glauber': 'glauber dynamics', 'simple': 'metropolis chain'}[key]
				)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax.set_yscale('log')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([35, 100, 500])
# ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
ax.set_xlabel('Steps')
# ax.set_ylabel('Reconstruction Error')
ax.set_ylabel('Energy')
plt.legend(by_label.values(), by_label.keys())
plt.grid()
plt.tight_layout()
plt.show()



