import numpy as np
from scipy.io import savemat

with open('output.txt') as f:
    X = [int(x) for x in f.readline().split()]
    savemat('answer_ChainTamers.mat', {'x_estimate':X}, appendmat=True, format='5', oned_as='column')
