# -*- coding: utf-8 -*-
import logging

import copy
from typing import List
from typing import Dict
from overrides import overrides
import tqdm

from allennlp.data.tokenizers import Token

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# from allennlp.data.dataset_readers.conll2003 import _is_divider
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TUPLE_ORDER = ['tokens', 'pos', 'ner']



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


@DatasetReader.register("word_tag_tuple_readers")
class WordTagTupleReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tuple_order: List = TUPLE_ORDER,
                 use_tag: str = "ner",
                 tuple_separator: str = "|",
                 token_separator: str = " ",
                 input_scheme: str = 'IOB1',
                 tag_scheme: str = 'IOB2',
                 ignore_tag: str = None,
                 encoding: str = "latin-1",
                 lm_task: bool= False,
                 start_end: bool= False,
                 lazy: bool = True) -> None:
        super(WordTagTupleReader, self).__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': TokenIndexer()}
        self._tuple_order = tuple_order
        self._use_tag = use_tag
        self._tuple_separator = tuple_separator
        self._token_separator = token_separator
        self._input_scheme = input_scheme
        self._tag_scheme = tag_scheme
        self._ignore_tag = ignore_tag
        self._encoding = encoding
        self._lm_task = lm_task
        self._start_end = start_end

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
            # Group into sentence.
            for line in tqdm.tqdm(data_file.readlines()):
                if not (line.strip() == ''):
                    fields = [wt.split(self._tuple_separator) for wt in line.split(self._token_separator)]
                    cols = [list(field) for field in zip(*fields)]

                    assert len(cols) == len(self._tuple_order)
                    col_header = zip(self._tuple_order, cols)
                    keyed_cols = dict([(key, list(field)) for key, field in col_header])
                    # print(keyed_cols)

                    assert "tokens" in keyed_cols
                    # TextField requires ``Token`` objects
                    for label in keyed_cols.keys():
                        if label != 'tokens':
                            keyed_cols[label] = [val.rstrip().upper() for val in keyed_cols[label]]
                    tokens = [Token(token) for token in keyed_cols["tokens"]]
                    sequence = TextField(tokens, self._token_indexers)
                    tags = keyed_cols[self._use_tag]
                    if self._input_scheme == 'IOB1' and self._tag_scheme == 'IOB2':
                        iob2(tags, ignore=self._ignore_tag)
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
    def from_params(cls, params: Params) -> 'WordTagTupleReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        tuple_order = params.pop('tuple_order', TUPLE_ORDER)
        use_tag = params.pop('use_tag', 'ner')
        tuple_separator = params.pop('tuple_separator', '|')
        token_separator = params.pop('token_separator', ' ')
        ignore_tag = params.pop('ignore_tag', None)
        encoding = params.pop('encoding', 'latin-1')
        lm_task = params.pop('lm_task', False)
        start_end = params.pop('start_end', False)
        params.assert_empty(cls.__name__)
        return WordTagTupleReader(token_indexers=token_indexers,
                                  tuple_order=tuple_order, use_tag=use_tag, 
                                  tuple_separator=tuple_separator, 
                                  token_separator=token_separator,
                                  ignore_tag=ignore_tag,
                                  encoding=encoding, lm_task=lm_task, 
                                  start_end=start_end)
