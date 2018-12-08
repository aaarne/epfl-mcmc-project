import os
from argparse import ArgumentParser


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input_dir', type=str, default='inputs',
        help='directory containing input files')

    args = argparser.parse_args()

