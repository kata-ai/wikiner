
import sys
import argparse
import json
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from ingredients.custom_conll import CustomConll
from ingredients.wpne_readers import WordTagTupleReader


def main(args):
    config = Params.from_file(args.config_file)
    reader_config = config.pop("dataset_reader")
    reader = DatasetReader.from_params(reader_config)
    x = reader.read(args.input_file)
    if args.output_file is None:
        output_file = sys.stdout
    else:
        output_file = open(args.output_file, mode='w')
    for instance in x:
        tokens = instance.fields['tokens'].tokens
        tags = instance.fields['tags'].labels
        text = [t.text for t in tokens]
        for token, tag in zip(text, tags):
            print(f'{token}\t{tag}', file=output_file)
        print('', file=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert dataset readers to jsonl for predict output'
    )
    parser.add_argument('--config-file', type=str,
                        help='config path of dataset readers', required=True)
    parser.add_argument('--input-file', type=str,
                        help='input path of multitags conll', required=True)
    parser.add_argument('--output-file', type=str,
                        help='output path of line json pred', default=None)
    merge_args = parser.parse_args()
    main(merge_args)
