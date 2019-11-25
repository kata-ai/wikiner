# -*- coding: utf-8 -*-
import logging
import itertools

import copy
from typing import List
from typing import Dict
from overrides import overrides
import tqdm

from allennlp.data.tokenizers import Token

from allennlp.common import Params
# from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.conll2003 import _is_divider
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

COLUMNS_HEADER = ['tokens', 'ner', 'srl']


def iob2(tags, ignore: str = None):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        if ignore and ignore in tag:
            tags[i] = 'O'
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


@DatasetReader.register("custom_conll")
class CustomConll(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 columns_header: List = COLUMNS_HEADER,
                 use_header: str = "ner",
                 ignore_tag: str = None,
                 input_scheme: str = 'IOB1',
                 tag_scheme: str = 'IOB2',
                 field_sep: str = None,
                 encoding: str = "latin-1",
                 lm_task: bool = False,
                 start_end: bool = False,
                 max_characters_per_token: int = 50,
                 lazy: bool = True) -> None:
        super(CustomConll, self).__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': TokenIndexer()}
        self._columns_header = columns_header
        self._use_header = use_header
        self._ignore_tag = ignore_tag
        self._input_scheme = input_scheme
        self._tag_scheme = tag_scheme
        self._field_sep = field_sep
        self._encoding = encoding
        self._lm_task = lm_task
        self._start_end = start_end
        self._max_characters_per_token = max_characters_per_token

    @staticmethod
    def _create_lm_task(words):
        targets_fwd = copy.copy(words[1:])
        targets_fwd.extend(['END'])
        targets_bwd = ['START']
        targets_bwd.extend(copy.copy(words[:-1]))
        return targets_fwd, targets_bwd

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r", encoding=self._encoding) as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in tqdm.tqdm(itertools.groupby(data_file, _is_divider)):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    # fields = [line.strip().split() for line in lines]
                    # if self._field_sep is not None:
                    fields = fields = [line.strip().split(self._field_sep) for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    cols = [list(field) for field in zip(*fields)]
                    assert len(cols) == len(self._columns_header)
                    col_header = zip(self._columns_header, cols)
                    keyed_cols = dict([(key, list(field)) for key, field in col_header])

                    assert "tokens" in keyed_cols
                    # TextField requires ``Token`` objects
                    for label in keyed_cols.keys():
                        if label != 'tokens':
                            keyed_cols[label] = [val.upper() for val in keyed_cols[label]]
                    tokens = [Token(token[:self._max_characters_per_token]) for token in keyed_cols["tokens"]]
                    sequence = TextField(tokens, self._token_indexers)

                    # tags = SequenceLabelField(keyed_cols[self._use_header], sequence)
                    tags = keyed_cols[self._use_header]
                    if self._input_scheme == 'IOB1' and self._tag_scheme == 'IOB2':
                        iob2(tags, ignore=self._ignore_tag)
                        tags = [tag if "B-" in tag or "I-" in tag or "O" in tag else "O" for tag in tags]
                    tags = SequenceLabelField(tags, sequence)

                    instance = {
                        'tokens': sequence, 'tags': tags
                    }
                    # for label in keyed_cols.keys():
                    #     if label != 'tokens' and label != 'srl':
                    #         instance[label] = SequenceLabelField(keyed_cols[label],
                    #                                              sequence, label)

                    # if self._lm_task and self._start_end:
                    #     fwd_lm, bwd_lm = self._create_lm_task(keyed_cols['tokens'])
                    #     instance['fwd_labels'] = SequenceLabelField(fwd_lm, sequence, 'fwd_labels')
                    #     instance['fwd_labels_mask'] = LabelField(1, 'task_mask', True)
                    #     instance['bwd_labels'] = SequenceLabelField(bwd_lm, sequence, 'bwd_labels')
                    #     instance['bwd_labels_mask'] = LabelField(1, 'task_mask', True)

                    yield Instance(instance)

    def text_to_instance(self, tokens: List[Token],
                         verb=None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, token_indexers=self._token_indexers)
        # print(sequence.sequence_length())
        instance = {'tokens': sequence}
        if verb:
            # logging.info(sequence.sequence_length())
            instance['verb_indicator'] = SequenceLabelField(verb, sequence)
        return Instance(instance)

    @classmethod
    def from_params(cls, params: Params) -> 'CustomConll':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        columns_header = params.pop('columns_header', COLUMNS_HEADER)
        use_header = params.pop('use_header', 'ner')
        encoding = params.pop('encoding', 'latin-1')
        ignore_tag = params.pop('ignore_tag', None)
        input_scheme = params.pop('input_scheme', 'IOB1')
        tag_scheme = params.pop('tag_scheme', 'IOB2')
        field_sep = params.pop('field_sep', None)
        lm_task = params.pop('lm_task', False)
        max_characters_per_token = params.pop('max_characters_per_token', 50)
        start_end = params.pop('start_end', False)
        params.assert_empty(cls.__name__)
        return CustomConll(token_indexers=token_indexers,
                           columns_header=columns_header,
                           use_header=use_header,
                           ignore_tag=ignore_tag,
                           input_scheme=input_scheme,
                           tag_scheme=tag_scheme,
                           field_sep=field_sep,
                           encoding=encoding,
                           lm_task=lm_task,
                           max_characters_per_token=max_characters_per_token, 
                           start_end=start_end)
