import os
import numpy as np
from scipy.io import loadmat
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


def transition(W, Y, X):
    '''
    Return a transition from the current state and its associated energy
    input:  W - features matrix
            Y - observations vector
            X - input vector
    output: the new state and its energy
    '''
    # Sample
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


def create_schedule(schedule_name, beta_0=0.1, beta_max=4, steps=60):
    '''
    Create a schedule for the simulated annealing parameter beta
    input:  schedule_name: one of {'linear', 'exponential', 'logarithmic'}
            beta_0: the smallest value of beta
            beta_max: the final value of beta
            steps: the amount of steps
    output: a function mapping k to the value of beta
    '''
    alpha_lin = (beta_max - beta_0) / steps
    alpha_exp = (beta_0 / beta_max) ** (1 / steps)
    alpha_log = ((beta_max / beta_0) - 1) / (np.log(1 + steps))

    return {
        'linear':       lambda k: beta_0 + alpha_lin * k,
        'exponential':  lambda k: beta_0 * alpha_exp ** (-k),
        'logarithmic':  lambda k: beta_0 * (1 + alpha_log * np.log(1 + k))
    }[schedule_name]



def mcmc(W, Y, ground_truth, seed=None, debug=False, method='simple', schedule_type='logarithmic'):
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
        'adaptive': mcmc_adaptive,
        'glauber': glauber_dynamics
    }[method](W, Y, ground_truth, seed, debug, schedule_type)


def mcmc_simple(W, Y, ground_truth, seed, debug, schedule_type):
    np.random.seed(seed)
    max_steps = 1000
    total_steps, step = 0, 0
    beta_0, beta_max = 0.1, 4
    k = 0
    schedule = create_schedule(schedule_type, beta_max=beta_max, beta_0=beta_0)
    beta = schedule(k)

    # Initialize the input vector
    min_X = X = init(W.shape[1], seed)
    min_energ = energ = energy(W, Y, X)
    error = reconstruction_error(min_X, ground_truth)

    # Initialize the statistics vector
    steps, betas, energies, errors = [], [], [energ], [error]
    single_energies = []

    # Minimize the energy
    while beta < beta_max:
        total_steps += 1
        step += 1
        # Compute a transition
        aux_X, aux_energ = transition(W, Y, X)

        # Compute the acceptance probability
        accept_prob = min(1, np.exp(-beta * (aux_energ - energ)))

        # Decide whether to do the transition or not
        if np.random.uniform() <= accept_prob:
            X, energ = aux_X, aux_energ
            # Update the minimum energy
            if energ < min_energ:
                if debug:
                    print(f'Reached smaller energy {energ} at step {total_steps} and beta {beta}. (Local steps: {step})')
                min_energ, min_X, step = energ, X, 0
                steps.append(total_steps)
                betas.append(beta)
                energies.append(min_energ)
                errors.append(reconstruction_error(min_X, ground_truth))

        single_energies.append(energ)

        # Perform the annealing if we reached a lower enough energy
        if step >= max_steps:
            step, X = 0, min_X
            k += 1
            mu =1 + (energ - min_energ) / energ
            beta = mu * schedule(k)
            if debug:
                print(f'Changed beta to {beta}')

    return min_X, steps, betas, energies, errors, single_energies


def mcmc_adaptive(W, Y, ground_truth, seed, debug, schedule_type):
    np.random.seed(seed)

    total_steps, step = 0, 0
    beta_0, beta_max = 0.1, 4
    beta = beta_0

    #Parameters
    alpha = 0.07
    L = 100
    N =10
    epsilon = 0.1
    n1 = 1 # N_over < n1 too fast
    n2 = 5 # N_over > n2 too slow

    # Initialize the input vector
    min_X = X = init(W.shape[1], seed)
    U_ref = min_energ = energ = energy(W, Y, X)
    error = reconstruction_error(min_X, ground_truth)

    # Initialize the statistics vector
    steps, betas, energies, errors = [], [], [energ], [error]
    single_energies = []

    # Minimize the energy
    while beta < beta_max:
        U_avg = 0
        N_over = 0 # counts how many sub steps of a cooling step (same beta) are above avarege Energy

        for k in range(N):
            U_mes = 0
            for l in range(L):
                total_steps += 1
                # Compute a transition
                aux_X, aux_energ = transition(W, Y, X)

                # Compute the acceptance probability
                accept_prob = min(1, np.exp(-beta * (aux_energ - energ)))

                # Decide whether to do the transition or not
                if np.random.uniform() <= accept_prob:
                    X, energ = aux_X, aux_energ
                    # Update the minimum energy
                    if energ < min_energ:
                        if debug:
                            print(f'Reached smaller energy {energ} at step {total_steps} and beta {beta}. (Local steps: {l})')
                        min_energ, min_X = energ, X
                        steps.append(total_steps)
                        betas.append(beta)
                        energies.append(min_energ)
                        errors.append(reconstruction_error(min_X, ground_truth))

                single_energies.append(energ)
                #calculate new avg energy
                U_mes += energ

            U_mes /= L
            U_avg += U_mes

            if U_mes > U_ref - epsilon:
                N_over += 1

        # Verify equilibrium of energy change
        t_eq = False
        if N_over >= n1: #equilibrium achieved
            t_eq = True

        t_slow = False
        if N_over >= n2: # cooling too slow
            t_slow = True
        if debug:
            print(f'U_ref {U_ref}, U_avg {U_avg/N}, t_eq {t_eq}, t_slow {t_slow}')
        U_ref = U_avg / N
        k+=1
        X = min_X
        if t_eq:
            if t_slow:
                #change inverse proportional to the difference between avg and minimal
                mu = (U_avg - min_energ) / U_avg
                beta = beta + alpha / mu
            else:
                beta = beta + alpha
        else: # cooling too fast, decrese alpha
            #change proportional to the difference between avg and minimal
            mu = (U_avg - min_energ) / U_avg
            beta = beta + alpha * mu




    return min_X, steps, betas, energies, errors, single_energies


def glauber_dynamics(W, Y, ground_truth, seed, debug, schedule_type):
    n = W.shape[1]
    min_X = X = init(n, seed)
    min_energ = energy(W, Y, X)
    error = reconstruction_error(min_X, ground_truth)

    max_steps = 1000
    total_steps, step = 0, 0
    beta_0, beta_max = 0.1, 4
    k = 0
    schedule = create_schedule(schedule_type, beta_max=beta_max, beta_0=beta_0)
    beta = schedule(k)

    steps, betas, energies, errors, single_energies = [], [], [], [], []

    def compute_probability(e_x, e_neg_x, sign):
        return (1 + sign * np.tanh(beta * (e_neg_x - e_x))) / 2

    def generate(x_0):
        x = x_0
        e = energy(W, Y, x)
        while True:
            # Select random index in [0...n-1]
            i = np.random.randint(n)

            # Generate the state with flipped sign at index i
            x_new, e_new = transition(W, Y, x)

            # compute vector of the probabilities
            p = [   compute_probability(e, e_new,  x[i]),
                    compute_probability(e, e_new, -x[i])]

            # choose x or x_tilde according to p
            if np.random.choice([+1, -1], p=p) != x[i]:
                x = x_new
                e = e_new

            yield x, e

    for x, energ in generate(X):
        single_energies.append(energ)
        step += 1
        total_steps += 1

        if energ < min_energ:
            if debug:
                print(f'Reached smaller energy {energ} at step {total_steps} and beta {beta}. (Local steps: {step})')
            min_energ, min_X, step = energ, x, 0
            steps.append(total_steps)
            betas.append(beta)
            energies.append(min_energ)
            errors.append(reconstruction_error(min_X, ground_truth))

        # Perform the annealing if we reached a lower enough energy
        if step >= max_steps:
            step = 0
            k += 1
            beta = schedule(k)
            if debug:
                print(f'Changed beta to {beta}')
            if beta >= beta_max:
                break

    return min_X, steps, betas, energies, errors, single_energies


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input', type=str, default='input.txt',
        help='input file containing features and observations')
    argparser.add_argument('--output', type=str, default='output.txt',
        help='output file where the energy evolution and the final prediction \
        will be stored')
    argparser.add_argument('--input_ref', type=str, default='input_vect.txt',
        help='file containing the ground truth vector')
    argparser.add_argument('--method', type=str, default='glauber',
        help='MCMC method. One of {simple, adaptive, glauber}')
    argparser.add_argument('--schedule', type=str, default='logarithmic',
        help='Schedule type for simulated annealing. One of {linear, exponential, logarithmic}')
    argparser.add_argument('--input_type', type=str, default='txt',
        help='type of input file')
    argparser.add_argument('--store_energy', type=str, default='True',
        help='whether to also store the energy evolution along with the final \
        prediction')

    args = argparser.parse_args()
    # Read features and observations
    if args.input_type == 'txt':
        with open(args.input, 'r') as f:
            f.readline()
            data = f.read().split('\n')
            W = np.array([[float(x) for x in d.split()] for d in data[:-1]])
            Y = np.array([float(x) for x in data[-1].split()])
    elif args.input_type == 'mat':
        data = loadmat(args.input)
        W, Y = data['W'], data['Y'].reshape(-1)

    # Also read the groundtruth if it exists
    ground_truth = np.ones(W.shape[1])
    if os.path.exists(args.input_ref):
        if args.input_type == 'txt':
            with open(args.input_ref, 'r') as f:
                ground_truth = np.array([int(x) for x in f.readline().split()])
        elif args.input_type == 'mat':
            ground_truth = loadmat(args.input_ref)['X'].reshape(-1)

    # Run the MCMC algorithm
    min_X, steps, betas, energies, errors, single_energies =\
        mcmc(W, Y, ground_truth,
            debug=True,
            method=args.method,
            schedule_type=args.schedule)

    # Write the data
    with open(args.output, 'w') as f:
        min_X_str = ' '.join([str(x) for x in min_X]) + '\n'
        f.write(min_X_str)
        f.write(str(energies[-1]))
        # Also store the energy evolution
        if args.store_energy == 'True':
            for i in range(len(steps)):
                s = f'{steps[i]} {betas[i]} {energies[i]} {errors[i]}\n'
                f.write(s)

    with open('output_energies.txt', 'w') as f:
        for energ in single_energies:
            f.write(f'{energ}\n')
