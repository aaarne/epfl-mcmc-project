# Markov Chains and Algorithmic Applications: Mini-Project

We use Markov Chain Monte Carlo techniques (a Metropolis chain and Glauber dynamics) combined with simulated annealing for a nonlinear optimization objective.

## Team members
- Gheorghe Druta
- Alexandru Mocanu 
- Arne Sachtler 

## Code organization
The code is organized in several modules:
1. Input generator -> `random_input_generator.py`
2. Nonlinear estimator algorithm -> `nonlinear_estimator.py`
3. Experiment runner -> `run_experiments.py`
4. Batch experiment runner -> `experiments.py`
5. Plotter -> `plot.py`

## Code description
### 1. Input generator
Parameters:

| parameter | description |
| --------- | ----------- |
| `m` | number of samples |
| `n` | number of features in the input vector |
| `out_mat_file` | file where to output the features matrix W and the observations vector Y |
| `out_in_file` | file where to output the real input vector |

Each of the features in W is drawn from a normal distribution of mean 0 and variance 1. X is constructed by extracting its elements uniformly in from {-1, 1}. Using W and X we now compute Y, output everything to the files passed as arguments or just return the matrices, in case those files are not specified. 

### 2. Nonlinear estimator
Parameters:

| parameter | description |
| --------- | ----------- |
| `input`| file containing W and Y|
| `output`| file where to write the evolution of the energy and the errors over the algorithm's steps, as well as the final estimated input vector|
| `input_ref`| file containing the real input vector|
| `method` | the algorithm to use. One of `simple`, `adaptive`, `glauber`|
| `schedule` | the simulated annealing schedule type. One of `linear`, `exponential`, `logarithmic`. Note this property holds for beta, not for the temperature.|

The important part in this script are the functions implementing the several algorithms for solving the problem. The `mcmc` function provides the option of selecting between three different algorithms: 
- a Metropolis-Hastings chain with naïve beta-scheduling, 
- a Metropolis-Hastings chain with adaptive beta-scheduling, 
- and Glauber dynamics. 
 
Moreover, we can also choose between three schedules for the simulated annealing: linear, exponential and logarithmic. Let us describe the MCMC algorithms:

- __simple Metropolis chain__:
We follow the Metropolis algorithm presented in the homework's description. For the annealing strategy, we adopt the following non-adaptive approach: we keep count of the moments we decrease the energy and we increase beta just in case we did not improve our estimation over more than a fixed number of steps (in our case 1000). The algorithm stops when we've reached a value of beta greater than or equal to a threhold value.
- __adaptive Metropolis chain__:
We tried an adaptive scheduling algorithm in order to adapt the rate of cooling (change of alpha), based on the dynamics of the avarage energy
The stategy is applied above another simple schedule type (the current implementation works only for the linear scheduling.
For each beta step we mesure `U_mes` energies of `N` substeps of length `L` iterations and compare it with `U_avg` calculated at the previous beta.
if none `U_mes` is bigger then we decrease `alpha`
if between 1 and 4 `U_mes` are bigger then we decrease consider the cooling to be in an equilibrium
if more than 5 `U_mes` are bigger then we increase `alpha`
- __Glauber dynamics__:
As well as for the Metropolis chain we follow the formulas in the project handout for the implementation. The `generate` generator creates an infinite stream of states and the respective energies. Then, we iterate over the stream in order to analyze the performance of the states and perform the beta-scheduling. By this approach we can keep the fundamental sampling of states and the simulated annealing strategy separated and combine the algorithms.

The helper functions used in the algorithms are:
- `init`: Gives an initial random estimation for the input vector.
- `energy`: Computes the energy given W, X, and Y.
- `transition`: Computes a uniformly random transition and its corresponding energy given W, X and Y.
- `reconstruction_error`: Computes the reconstruction error given an estimation and the real input vector.
- `create_schedule`: Creates a schedule for annealing beta, either linear, exponential or logarithmic in terms of the number of steps of annealing.

### 3. Experiment runner
Parameters:

| parameter | description |
| --------- | ----------- |
| `alpha`| the ratio between the number of samples and the number of features|
| `n`| the number of features of the input vector|
| `num_runs`| the number of runs of the algorithm used for computing the expected value and the variance of the reconstruction error|
| `output`| file where to write the results|

The point of this script is implementing the `run_experiment` function, which for each of the `num_runs` steps generates a set of inputs W, X, Y and calls the `mcmc` function in the `nonlinear_estimator.py` script to compute the reconstruction error. These errors are collected in an array and they are used for computing the expected value and the standard deviation of the error. We compute these statistics for both the last estimated input vectors as well as for the ones that minimized the reconstruction error in each run. These results are returned by our function and are also stored in the output file, in case it is provided.

### 4. Batch experiment runner
This script simply calls the `run_experiment` function from the script presented above for alphas ranging between 0.01 and 1.4 with a step of 0.01 between them, for 100 features and 10 runs of the algorithm for each alpha.

### 5. Plotter
Parameters:

| parameter | description |
| --------- | ----------- |
| `data`| file containing the energies, errors and final estimated input vector|
| `data_energies`| file containing the energies for each state visited while running the algorithm|
| `input_ref`| file containing the real input vector|
| `plot_dir`| directory where to store the plot(s)|
| `plot_alpha`| whether to do a plot for the reconstruction error in terms of alpha or not. Possible values are 'no_alpha', in case we don't plot the reconstruction error in terms of alpha, 'last', in case we use the last estimated input vector, 'min', in case we use the estimated input vector for which we obtained the smallest reconstruction error, or 'both', if we want a plot combining the ones for 'last' and 'min'.|

The script provides three plotting functions:
- `plot_evolution`: Plot the energy or the reconstruction error as a function of time.
- `plot_alpha`: Plot the dependency of the reconstruction error on alpha.
- `plot_energies`: Plot the energy in terms of the step, but this time with a logarithmic scale on y.
