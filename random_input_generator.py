import numpy as np
from argparse import ArgumentParser

def np_to_string(mat):
    '''
    Transform a matrix to a string representation to be written to a file
    input:  mat - matrix to be converted to string
    output: string representation of the matrix
    '''
    s = f"{mat.shape[0]} {mat.shape[1]}\n"
    for row in mat:
        row = ' '.join(list(map(lambda x: str(x), row)))
        s += row + '\n'
    return s

def generate_input(m, n, out_mat_file, out_in_file, seed=None):
    '''
    Generate the input matrix, observations and real input values
    input:  m - number of samples
            n - number of features
            out_mat_file - file where to output the matrix and the observations.
            If None, don't write anything to files
            out_in_file - file where to output the real input values
            seed - random seed
    output: W - matrix of features
            X - input vector
            Y - observed values
    '''
    np.random.seed(seed)
    # Generate the matrix
    W = np.random.normal(0, 1, (m, n))
    # Generate a random input vector
    X = np.random.choice([-1, 1], n)
    # Compute the observations
    Y = np.maximum(0, W.dot(X) / np.sqrt(n))

    if out_mat_file is not None:
        # Transform matrix in string format
        mat = np_to_string(W)

        # Write matrix to file
        f = open(out_mat_file, 'w')
        f.write(mat)

        # Write observation to file
        f.write(' '.join(list(map(lambda x: str(x), Y))))
        f.close()

        # Write the input vector
        f = open(out_in_file, 'w')
        f.write(' '.join(list(map(lambda x: str(x), X))))
        f.close()

    return W, X, Y


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-m', type=int, default=10, help='number of samples')
    argparser.add_argument('-n', type=int, default=10, help='number of features')
    argparser.add_argument('--out_mat_file', type=str, default='input.txt',
        help='output file for the matrix and the observations')
    argparser.add_argument('--out_in_file', type=str, default='input_vect.txt',
        help='output file for the input vector')

    args = argparser.parse_args()

    generate_input(args.m, args.n, args.out_mat_file, args.out_in_file)
