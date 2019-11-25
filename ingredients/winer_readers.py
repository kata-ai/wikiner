# -*- coding: utf-8 -*-
import os
import logging
import itertools
import random
import copy
from typing import List
from typing import Dict
from typing import NamedTuple
import tarfile
import warnings
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

LABEL_MAP = ['PER', 'LOC', 'ORG', 'MISC']

# Span = namedtuple('Span' , 'sent_idx start end label')
class Span(NamedTuple):
    sent_idx: int
    start: int
    end: int
    label: int
    value: str = None

# SentLabel = namedtuple('SentLabel', 'sentence tag')
class SentLabel(NamedTuple):
    sentence: List[str]
    tag: List[str]

# Doc = namedtuple('Wikidoc', 'sentences')
class Doc():
    def __init__(self, doc_id: int, sentences: List[SentLabel]):
        self.doc_id: int = doc_id
        self.sentences: List[SentLabel] = sentences

def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        # if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
        if first_token == "ID":
            return True
        else:
            return False

@DatasetReader.register("targz_readers")
class TargzReaders(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 vocab_file: str = None,
                 mentions_tarfile: str = None,
                 compression_mode: str = 'gz',
                 label_map: List[str] = None,
                 encoding: str = "utf-8",
                 lm_task: bool= False,
                 start_end: bool= False,
                 lazy: bool = True) -> None:
        super(TargzReaders, self).__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': TokenIndexer()}
        self._mentions_tarfile = mentions_tarfile
        self._compression_mode = compression_mode
        self._vocab_file = vocab_file
        self._encoding = encoding
        self._lm_task = lm_task
        self._start_end = start_end
        self._label_map = label_map

        self.vocab = []
        self.load_vocab()

    @staticmethod
    def _create_lm_task(words):
        targets_fwd = copy.copy(words[1:])
        targets_fwd.extend(['END'])
        targets_bwd = ['START']
        targets_bwd.extend(copy.copy(words[:-1]))
        return targets_fwd, targets_bwd

    def load_vocab(self):
        with open(self._vocab_file, mode='r', encoding=self._encoding) as vocab:
            for word in tqdm.tqdm(vocab.readlines()):
                self.vocab.append(word.split()[0])

    def process_documents(self, content):
        # documents = {}
        documents_obj = {}
        curr_id = -1
        for is_divider, lines in tqdm.tqdm(itertools.groupby(content, _is_divider)):
            # Ignore the document divider chunks, so that `lines` corresponds to the 
            # a single sentence.
            for line in lines:
                line = line.rstrip('\n')
                tokens = line.split()
                if tokens[0] == 'ID' and is_divider:
                    curr_id = tokens[1]
                    if curr_id in documents_obj:
                        warnings.warn(f'duplicate {curr_id}')
                    else:
                        # documents[curr_id] = []
                        documents_obj[curr_id] = Doc(curr_id, [])
                elif not (line.strip() == ''):
                    tokens = [Token(self.vocab[int(idx)]) for idx in tokens]
                    # documents[curr_id].append(tokens)
                    sent = SentLabel(tokens, ['O'] * len(tokens))
                    documents_obj[curr_id].sentences.append(sent)
        return documents_obj

    def process_mentions(self, mentions, documents: Dict[str, Doc]):
        curr_id = -1
        for is_divider, lines in tqdm.tqdm(itertools.groupby(mentions, _is_divider)):
            # Ignore the document divider chunks, so that `lines` corresponds to the 
            # a single sentence.
            for line in lines:
                line = line.rstrip('\n')
                tokens = line.split()
                if tokens[0] == 'ID' and is_divider:
                    curr_id = tokens[1]
                    if curr_id in documents:
                        # warnings.warn(f'duplicate {curr_id}')
                        pass
                    else:
                        # documents_label[curr_id] = []
                        documents[curr_id] = Doc(curr_id, [])
                elif not (line.strip() == ''):
                    # construct span object
                    span = Span(*tokens)
                    sent_idx = int(span.sent_idx)
                    start = int(span.start)
                    label = span.label
                    end = int(span.end)
                    # create array of I-{label} to fill array value
                    tag_span = [f'I-{self._label_map[int(label)]}'] * len(range(start, end))
                    # change first index span as B-{label}
                    tag_span[0] = f'B-{self._label_map[int(label)]}'
                    documents[curr_id].sentences[sent_idx].tag[start:end] = tag_span

        return documents

    @overrides
    def _read(self, file_path):
        """
        filepath : str
            path refers to Documents.tar.gz file
        """
        file_path = cached_path(file_path)
        doc_tar = tarfile.open(file_path, f"r:{self._compression_mode}")
        print(doc_tar.getnames())
        ent_tar = tarfile.open(self._mentions_tarfile, f"r:{self._compression_mode}")
        print(ent_tar.getnames())
        ent_base = ent_tar.getnames()[0].split('/')[0]
        for doc_member in doc_tar.getnames():
            # print(doc_member)
            doc = doc_tar.extractfile(doc_member)
            if doc is not None:
                content = doc.readlines()
                content = [line.decode(self._encoding) for line in content]

                documents_obj = self.process_documents(content)

                # ent_name = os.path.join(ent_base, doc_member.split('/')[-1])
                ent_name = f"{ent_base}/{doc_member.split('/')[-1]}"
                print(doc_member)
                print(ent_name)
                ent = ent_tar.extractfile(ent_name)
                if ent:
                    mention = ent.readlines()
                    mention = [line.decode(self._encoding) for line in mention]
                    documents_obj = self.process_mentions(mention, documents_obj)

                for _, doc in tqdm.tqdm(documents_obj.items()):
                    for sent_tag in doc.sentences:
                        # print(len(sent_tag.sentence))
                        # print(len(sent_tag.tag))
                        sequence = TextField(sent_tag.sentence, self._token_indexers)
                        tags = SequenceLabelField(sent_tag.tag, sequence)
                        instance = {
                            'tokens': sequence, 'tags': tags
                        }
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
    def from_params(cls, params: Params) -> 'TargzReaders':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        vocab_file = params.pop('vocab_file')
        mentions_tarfile = params.pop('mentions_tarfile')
        compression_mode = params.pop('compression_mode', 'gz')
        encoding = params.pop('encoding', 'utf-8')
        start_end = params.pop('start_end', False)
        label_map = params.pop('label_map', LABEL_MAP)
        lm_task = params.pop('lm_task', False)
        params.assert_empty(cls.__name__)
        return TargzReaders(token_indexers=token_indexers, vocab_file=vocab_file,
                            mentions_tarfile=mentions_tarfile, 
                            compression_mode=compression_mode, label_map=label_map, 
                            encoding=encoding,lm_task=lm_task, start_end=start_end)
        