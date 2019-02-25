import argparse
import sys
from collections import Counter

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='-', nargs='?',
                        help='Input file or - for standard input')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file. Default is standard output')
    parser.add_argument('-chars', action='store_true',
                        help='Create a character vocabulary')
    parser.add_argument('-lower', action='store_true',
                        help='Lowercase the input')
    parser.add_argument('-no_progress', action='store_true',
                        help='Disable the progress bar')
    args = parser.parse_args()

    input_stream = sys.stdin
    if args.input != '-':
        input_stream = open(args.input)
    output_stream = sys.stdout
    if args.output is not None:
        output_stream = open(args.output, 'w')

    counter = Counter()
    for line in tqdm(input_stream, disable=args.no_progress):
        line = line.rstrip()
        if args.lower:
            line = line.lower()
        if not args.chars:
            line = line.split()
        counter.update(line)

    if args.input != '-':
        input_stream.close()

    output_stream.writelines("{} {}\n".format(item, count) for item, count in counter.items())

    if args.output is not None:
        output_stream.close()

