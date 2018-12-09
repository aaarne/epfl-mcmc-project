import numpy as np
from argparse import ArgumentParser


def init(n, seed=None):
    '''
    Initialize the input vector
    input:  n - number of elements in the input vector
    output: an input vector with elements generated from a uniform distribution
    in range (0,1)
    '''
    np.random.seed(seed)
    return np.random.choice([-1, 1], n)


def energy(W, Y, X):
    '''
    Compute the energy function given the features, observations and input
    input:  W - features matrix
            Y - observations vector
            X - input vector
    output: the energy associated with the inputs
    '''
    pred = np.array([max(0, x) for x in W.dot(X)])
    e = Y - pred / np.sqrt(X.shape[0])
    return e.dot(e)


def transition(W, Y, X, seed=None):
    '''
    Return a transition from the current state and its associated energy
    input:  W - features matrix
            Y - observations vector
            X - input vector
    output: the new state and its energy
    '''
    # Sample
    np.random.seed(seed)
    ind = np.random.choice(X.shape[0])
    new_X = X.copy()
    new_X[ind] = -new_X[ind]
    return new_X, energy(W, Y, new_X)


def reconstruction_error(x, ground_truth):
    '''
    Compute the reconstruction error for a given estimation
    input:  x - estimated value of the input vector
            ground_truth - real value of the input vector
    output: the reconstruction error
    '''
    n = x.shape[0]
    e = x - ground_truth
    return e.dot(e) / (4 * n)


def mcmc(W, Y, ground_truth, seed=None, debug=False, annealing='simple'):
    '''
    Run the MCMC algorithm to find the input vector
    input:  W - features matrix
            Y - observations vector
            ground_truth - the real input vector
            seed - random seed
            debug - whether to print info while running the algorithm
            annealing - annealing strategy
    output: an estimation of the input vector along with the evolution of the
    energy
    '''
    return {
        'simple': mcmc_simple,
    }[annealing](W, Y, ground_truth, seed=seed, debug=debug)


def mcmc_simple(W, Y, ground_truth, seed=None, debug=False):
    max_steps = 1000
    total_steps, step = 0, 0
    beta, beta_step, beta_max = 0.1, 0.1, 4

    # Initialize the input vector
    np.random.seed(seed)
    min_X = X = init(W.shape[1], seed)
    min_energ = energ = energy(W, Y, X)
    error = reconstruction_error(min_X, ground_truth)

    # Initialize the statistics vector
    steps, betas, energies, errors = [], [], [energ], [error]

    # Minimize the energy
    while True:
        total_steps += 1
        step += 1
        # Compute a transition
        aux_X, aux_energ = transition(W, Y, X, seed)

        # Compute the acceptance probability
        accept_prob = min(1, np.exp(-beta * (aux_energ - energ)))

        # Decide whether to do the transition or not
        if np.random.uniform() <= accept_prob:
            X, energ = aux_X, aux_energ
            # Update the minimum energy
            if energ < min_energ:
                if debug:
                    print(f'Reached smaller energy {energ} at step \
{total_steps} and beta {beta}')
                min_energ, min_X, step = energ, X, 0
                steps.append(total_steps)
                betas.append(beta)
                energies.append(min_energ)
                errors.append(reconstruction_error(min_X, ground_truth))

        # Perform the annealing if we reached a lower enough energy
        if step >= max_steps:
            step, X = 0, min_X
            beta += beta_step
            if debug:
                print(f'Changed beta to {beta}')
            if beta >= beta_max:
                break

    return min_X, steps, betas, energies, errors


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input', type=str, default='input.txt',
        help='input file containing features and observations')
    argparser.add_argument('--output', type=str, default='output.txt',
        help='output file where the energy evolution and the final prediction \
        will be stored')
    argparser.add_argument('--input_ref', type=str, default='input_vect.txt',
        help='file containing the ground truth vector')

    args = argparser.parse_args()
    # Read features and observations
    f = open(args.input, 'r')
    f.readline()
    data = f.read().split('\n')
    W = np.array([[float(x) for x in d.split()] for d in data[:-1]])
    Y = np.array([float(x) for x in data[-1].split()])
    f.close()

    with open(args.input_ref, 'r') as f:
        ground_truth = np.array([int(x) for x in f.readline().split()])

    # Run the MCMC algorithm
    min_X, steps, betas, energies, errors = mcmc(W, Y, ground_truth, debug=True)

    # Write the data
    f = open(args.output, 'w')
    min_X_str = ' '.join([str(x) for x in min_X]) + '\n'
    f.write(min_X_str)
    for i in range(len(steps)):
        s = f'{steps[i]} {betas[i]} {energies[i]} {errors[i]}\n'
        f.write(s)
    f.close()
