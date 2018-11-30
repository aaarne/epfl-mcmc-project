import numpy as np
from argparse import ArgumentParser


def np_to_string(mat):
    s = f"{mat.shape[0]} {mat.shape[1]}\n"
    for row in mat:
        row = ' '.join(list(map(lambda x: str(x), row)))
        s += row + '\n'
    return s


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-m', type=int, default=10, help='number of samples')
    argparser.add_argument('-n', type=int, default=10, help='number of features')
    argparser.add_argument('--out_mat_file', type=str, default='input.txt',
        help='output file for the matrix and the observations')
    argparser.add_argument('--out_in_file', type=str, default='input_vect.txt',
        help='output file for the input vector')

    args = argparser.parse_args()

    # Generate the matrix
    W = np.random.normal(0, 1, (args.m, args.n))

    # Transform matrix in string format
    mat = np_to_string(W)

    # Write matrix to file
    f = open(args.out_mat_file, 'w')
    f.write(mat)

    # Generate a random input vector
    X = np.random.choice([-1, 1], args.n)

    # Compute the observations
    Y = np.maximum(0, W.dot(X) / np.sqrt(args.n))

    # Write observation to file
    f.write(' '.join(list(map(lambda x: str(x), Y))))
    f.close()

    # Write the input vector
    f = open(args.out_in_file, 'w')
    f.write(' '.join(list(map(lambda x: str(x), X))))
    f.close()
