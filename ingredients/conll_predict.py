import argparse
from contextlib import ExitStack
import json
import sys
from typing import Optional, IO, Dict

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from ingredients.commons import word_tokenize

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'srl': 'semantic-role-labeling',
        'decomposable_attention': 'textual-entailment',
        'bidaf': 'machine-comprehension',
        'bidaf-ensemble': 'machine-comprehension',
        'simple_tagger': 'sentence-tagger',
        'crf_tagger': 'sentence-tagger',
        'coref': 'coreference-resolution',
        'constituency_parser': 'constituency-parser',
}


class ConllPredict(Subcommand):
    def __init__(self, predictor_overrides: Dict[str, str] = {}) -> None:
        # pylint: disable=dangerous-default-value
        self.predictors = {**DEFAULT_PREDICTORS, **predictor_overrides}

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')

        subparser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')
        
        subparser.add_argument('--tokenizer', type=str, help='tokenizer, default None', default=None)

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')
        batch_size.add_argument('--batch_size', type=int, help=argparse.SUPPRESS)

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
        cuda_device.add_argument('--cuda_device', type=int, help=argparse.SUPPRESS)

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=_predict)

        return subparser

def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)
    return Predictor.from_archive(archive, 'tokenized-tagger')


def _run(predictor: Predictor,
         input_file: IO,
         output_file: Optional[IO],
         batch_size: int,
         print_to_console: bool,
         tokenizer: str = None) -> None:

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            result = predictor.predict_json(batch_data[0])
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data)

        count = 0
        for model_input, output in zip(batch_data, results):
            # string_output = json.dumps(output)
            if print_to_console:
                # print("input: ", model_input)
                # print("prediction: ", string_output)
                if isinstance(model_input['sentence'], str):
                    sent_list = word_tokenize(model_input['sentence'], ' ')
                else:
                    sent_list = model_input['sentence']
                # print(len(sent_list))
                # print(len(output['tags']))
                for i, token in enumerate(sent_list):
                    tag_ = [output[k][i] for k in sorted(output) if 'tags' in k]
                    # print(tag_)
                    print(token+"\t"+"\t".join(tag_), file=output_file)
                print('', file=output_file)
            count += 1
            # if output_file:
            #     output_file.write(string_output + "\n")

    batch_json_data = []
    for line in input_file:
        if not line.isspace():
            # Collect batch size amount of data.
            json_data = json.loads(line)
            # json_data['sentence'] = ' '.join(word_tokenize(json_data['sentence']))
            json_data['sentence'] = word_tokenize(json_data['sentence'], ' ')
            # print(json_data['sentence'])
            batch_json_data.append(json_data)
            if len(batch_json_data) == batch_size:
                _run_predictor(batch_json_data)
                batch_json_data = []

    # We might not have a dataset perfectly divisible by the batch size,
    # so tidy up the scraps.
    if batch_json_data:
        _run_predictor(batch_json_data)

def _predict(args: argparse.Namespace) -> None:
    print(args)
    predictor = _get_predictor(args)
    output_file = None

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)  # type: ignore
        if args.output_file:
            output_file = stack.enter_context(args.output_file)  # type: ignore

        _run(predictor,
             input_file,
             output_file,
             args.batch_size,
             not args.silent,
             args.tokenizer)
