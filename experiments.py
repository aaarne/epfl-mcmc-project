import os
import numpy as np
import pandas as pd
from run_experiments import run_experiment


if __name__ == '__main__':
    alphas = np.linspace(0.01, 1.4, 140)
    summary = {
        'mean_err': [],
        'std_err': [],
        'mean_min_err': [],
        'std_min_err': []
    }
    for alpha in alphas:
        print(f'Running for alpha {alpha}')
        mean_err, std_err, mean_min_err, std_min_err = \
            run_experiment(alpha, 100, 10)
        summary['mean_err'].append(mean_err)
        summary['std_err'].append(std_err)
        summary['mean_min_err'].append(mean_min_err)
        summary['std_min_err'].append(std_min_err)
        print(f'The mean error is {mean_err} and its standard deviation is {std_err}')
        print(f'The mean of the minimum errors is {mean_min_err} and their standard deviation is {std_min_err}')
        print()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('simple_annealing.csv')
