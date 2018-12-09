import os
import numpy as np
import time
from argparse import ArgumentParser
from nonlinear_estimator import mcmc
from random_input_generator import generate_input


def run_experiment(alpha, n, num_runs, output=None, seed=None):
    '''
    Run an experiment to determine the expected value and the standard deviation
    of the reconstruction error for a given number of features and samples
    input:  alpha - ratio between the number of samples and the number of
            features
            n - number of features
            num_runs - number of experiments to run for the estimation
            seed - random seed to be used
            output - name of the file where to write the expected value and the
            standard deviation. If None, don't write anything to a file
    output: expected value and standard deviation for the given setting
    '''
    errors = []
    min_errors = []
    for run in range(num_runs):
        start = time.time()

        # Generate input
        W, X, Y = generate_input(int(alpha * n), n, None, None)

        # Run the MCMC algorithm and get the reconstruction error
        _, _, _, _, errs = mcmc(W, Y, X)
        # Keep the minimum reconstruction error obtained
        errors.append(errs[-1])
        min_errors.append(np.min(errs))

        print(f'Run {run+1} of {num_runs} took {time.time()-start:.2f} seconds.')

    mean_err = np.mean(errors)
    std_err = np.std(errors)
    mean_min_err = np.mean(min_errors)
    std_min_err = np.std(min_errors)

    if output is not None:
        f = open(output, 'w')
        f.write(f'{mean_err} {std_err} {mean_min_err} {std_min_err}')

    return mean_err, std_err, mean_min_err, std_min_err


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--alpha', type=float, default=1,
        help='ratio between the number of samples and the number of features')
    argparser.add_argument('--n', type=int, default=1000,
        help='number of features for the input vector')
    argparser.add_argument('--num_runs', type=int, default=100,
        help='number of times to run the experiment')
    argparser.add_argument('--output', type=str, default=None,
        help='file where to output the expected error and standard devs')

    args = argparser.parse_args()
    mean_err, std_err, mean_min_err, std_min_err = run_experiment(args.alpha, args.n, args.num_runs, args.output)
    print(f'The mean error is {mean_err} and its standard deviation is {std_err}')
    print(f'The mean of the minimum errors is {mean_min_err} and their standard deviation is {std_min_err}')
