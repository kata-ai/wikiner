
import sys
import argparse
import json
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from ingredients.custom_conll import CustomConll


def main(args):
    config = Params.from_file(args.config_file)
    reader_config = config.pop("dataset_reader")
    reader = DatasetReader.from_params(reader_config)
    x = reader.read(args.input_file)
    if args.output_file is None:
        outfile = sys.stdout
    else:
        outfile = open(args.output_file, mode='w')
    # print(len(x.instances))
    for instance in x:
        tokens = instance.fields['tokens'].tokens
        text = ' '.join([t.text for t in tokens])
        line = str(json.dumps({"sentence": text}))
        # print(tokens)
        print(line, file=outfile)


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
