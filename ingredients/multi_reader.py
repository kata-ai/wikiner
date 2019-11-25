# -*- coding: utf-8 -*-
import logging

import json
from typing import Dict
import random
from collections import defaultdict
from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# from allennlp.data.dataset_readers.conll2003 import _is_divider
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import SequenceLabelField

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


@DatasetReader.register('multi-corpus-reader')
class MultiCorpusReader(DatasetReader):
    """
    corpus_readers: Dict of Sequence labeling dataset reader with different sources 
                    e.g Corpus A and B
                    in addition it also label word src wether the instance comes from corpus A or B
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 corpus_readers: Dict[str, DatasetReader] = defaultdict,
                 corpus_langmap: Dict[str, int] = None,
                 shuffle_corpus: bool = True,
                 lazy: bool = True
                ):
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.corpus_langmap = corpus_langmap
        self._corpus_readers = corpus_readers
        self._shuffle_corpus = shuffle_corpus
        for name, _ in self._corpus_readers.items():
            self._corpus_readers[name]._token_indexers = self._token_indexers


    @overrides
    def _read(self, file_path):
        """
        file_path: dict of corpus_reader keyname and filepath value
        """
        # file_path = cached_path(file_path)
        logging.info('yielding path %s', file_path)
        # if isinstance(file_path, Params):
        if isinstance(file_path, str):
            file_path = json.loads(file_path)
        file_paths = {} 
        for name, path in file_path.items():
            logger.info('name : %s, path : %s', name, path)
            file_paths[name] = self._corpus_readers[name].read(path)
            # for sent in self._corpus_readers[name].read(path):
            #     yield sent
        
        if self._shuffle_corpus:
            dataset = [self.generate_instance(sent, name) for name, corpus_gen in file_paths.items() \
                        for sent in corpus_gen]
            random.shuffle(dataset)
            for instance in dataset:
                yield instance
        else:
            # naive generator call, ordered by first corpus and then next corpus
            for name, corpus_gen in file_paths.items():
                logging.info('yielding dataset %s', name)
                for sent in corpus_gen:
                    yield self.generate_instance(sent, name)
    
    def generate_instance(self, sent, name=None):
        if self.corpus_langmap and name in self.corpus_langmap:
            lang = self.corpus_langmap[name]
            tokens = sent.fields['tokens']
            lang_seq = [lang] * tokens.sequence_length()
            lang_seq = SequenceLabelField(lang_seq, tokens, 'lang_tags')
            new_sent = {
                'tokens': tokens,
                'tags': sent.fields['tags'],
                'lang_indicator': lang_seq
            }
            return Instance(new_sent)
        else:
            return sent

    @classmethod
    def from_params(cls, params: Params) -> 'MultiCorpusReader':
        token_indexers_params = params.pop('token_indexers', {})
        token_indexers = TokenIndexer.dict_from_params(token_indexers_params)
        corpus_langmap = params.pop('corpus_langmap', None)
        logger.info('corpus langmap %s', corpus_langmap)
        shuffle_corpus = params.pop('shuffle_corpus', True)
        corpus_readers_params: Dict = params.pop('corpus_readers', {})
        corpus_readers = defaultdict()
        for name, params in corpus_readers_params.items():
            params['token_indexers'] = token_indexers_params
            choice = params.pop_choice('type', DatasetReader.list_available())
            corpus_readers[name] = DatasetReader.by_name(choice).from_params(params)
            # corpus_readers[name] = DatasetReader.from_params(**params)
        lazy = params.pop('lazy', True)
        params.assert_empty(cls.__name__)
        return MultiCorpusReader(token_indexers=token_indexers,
                                 corpus_readers=corpus_readers,
                                 corpus_langmap=corpus_langmap,
                                 shuffle_corpus=shuffle_corpus,
                                 lazy=lazy)
    
