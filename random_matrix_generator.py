import numpy as np
from argparse import ArgumentParser


def np_to_string(mat):
    s = ""
    for row in mat:
        row = ' '.join(list(map(lambda x: str(x), row)))
        s += row + '\n'
    return s


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-m', type=int, default=10, help='number of samples')
    argparser.add_argument('-n', type=int, default=10, help='number of features')
    argparser.add_argument('--out_file', type=str, default='mat.txt',
        help='output file for the matrix')

    args = argparser.parse_args()

    # Generate the matrix
    mat = np.random.normal(0, 1, (args.m, args.n))

    # Transform matrix in string format
    mat = np_to_string(mat)

    # Save matrix to file
    f = open(args.out_file, 'w')
    f.write(mat)
    f.close()
