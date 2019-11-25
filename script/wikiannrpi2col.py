
import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""convert wikiann rpi dataset format into two column conll""")
    parser.add_argument('filepattern', metavar='FILE', help='path to the corpus file')
    parser.add_argument('-o', '--output-dir', metavar='DIR', default=None,
                        help='output directory (default: {})'.format(None))
    parser.add_argument('--encoding', default='utf-8', help='file encoding (default: utf-8)')
    args = parser.parse_args()
    main(args)
