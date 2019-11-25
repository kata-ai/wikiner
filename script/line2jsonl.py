import argparse
import codecs
import re
import json

def word_tokenize(sentence, sep=r'(\W+)?'):
    return [x.strip() for x in re.split(sep, sentence) if x.strip()]

def print_jsonlfile(data, outfile):
    for line in data.readlines():
        line = line.rstrip('\n')
        line = line.rstrip(' ')
        line = line.rstrip('.')
        line = ' '.join(word_tokenize(line))
        json_line = str(json.dumps({'sentence': f'{line}.'}))
        print(json_line, file=outfile)

def main(args):
    data = codecs.open(args.input_file, mode='r', encoding=args.encoding)
    outfile = open(args.output_file, mode='w')
    print_jsonlfile(data, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert text line to jsonl for model output'
    )
    parser.add_argument('--input-file', type=str,
                        help='input path of multitags conll', required=True)
    parser.add_argument('--output-file', type=str,
                        help='output path of line json pred', default=None)
    parser.add_argument('--encoding', default='utf-8',
                        help='file encoding (default: utf-8)')
    merge_args = parser.parse_args()
    main(merge_args)
