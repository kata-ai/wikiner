import re
from collections import defaultdict

import numpy as np
from nltk.tokenize import RegexpTokenizer

class IOBTagger:
    O_TAG = 'O'
    BEGIN_PREFIX = 'B'
    INSIDE_PREFIX = 'I'
    DELIMITER = '-'

    def __init__(self, tokenizer=None) -> None:
        if tokenizer is None:
            tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        self.tokenizer = tokenizer

    def tag_uninteresting_span(self, span):
        return ((word, self.O_TAG) for word in self.tokenizer.tokenize(span))

    def tag_span_with(self, span, tag):
        words = self.tokenizer.tokenize(span)
        if not words:
            # yield from []
            for x in []:
                yield x
        else:
            yield (words[0], '{}{}{}'.format(self.BEGIN_PREFIX, self.DELIMITER, tag))
            # yield from ((word, '{}{}{}'.format(self.INSIDE_PREFIX, self.DELIMITER, tag))
            #             for word in words[1:])
            for x in ((word, '{}{}{}'.format(self.INSIDE_PREFIX, self.DELIMITER, tag))
                      for word in words[1:]):
                yield x

    @classmethod
    def normalize_tag(cls, tag):
        if len(tag) > 1 and tag[0] in [cls.BEGIN_PREFIX, cls.INSIDE_PREFIX] and \
           tag[1] == cls.DELIMITER:
            return tag[2:]
        else:
            return tag

def match_exact(span_a, span_b):
    return span_a['span_idx'] == span_b['span_idx']


def get_remaining(arr, idx):
    arr = np.array(arr)
    mask = np.ones(len(arr), np.bool)
    mask[idx] = False
    return arr[mask]


def create_span(sentence, tag_list, start_seg='B-', ignore_label='V'):
    """
    Create a list of span dict. It accept single sentence
    :param sentence: list of tag in string ['O', 'O', 'B-PER', 'O']
    :param tag_list: list of entity without O ['PER', 'LOC', 'EMAIL']
    :param start_seg: how the begin tag is prefixed
    :param ignore_label: string label which is ignored
    :return: list of span dict. Example:
    [
        {'span_idx': [2, 3], 'tag_idx': 0},
        {'span_idx': [5], 'tag_idx': 1},
        {'span_idx': [6, 7], 'tag_idx': 2}
    ]
    span_idx is the index of the sentence where the span spans
    tag_idx is the tag of that span
    """
    spans = []
    iob_tag_list = [('B-' + tag, 'I-' + tag) for tag in tag_list]
    status = 0  # flag to check if we are inside a span
    for sent_idx, tag in enumerate(sentence):
        if tag == 'O' or ignore_label in tag:
            status = 0
            continue
        stripped_tag = tag.split('-')[1]
        tag_idx = tag_list.index(stripped_tag)
        iob_tag = iob_tag_list[tag_idx]
        # the following code adopts the implementation of Fariz
        if str(tag) in iob_tag:
            if tag.startswith(start_seg):
                # we detect the start of a span. create span dict
                status = 1
                spans.append({'span_idx': [sent_idx], 'tag_idx': tag_idx})
            else:
                if status == 1:
                    # we detect a continuation of existing span
                    # get the last span, append idx to the span index
                    spans[-1]['span_idx'].append(sent_idx)
                else:
                    # we detect the start of a span, but starts with I-
                    # TODO should we detect this as span?
                    status = 1
                    spans.append({'span_idx': [sent_idx], 'tag_idx': tag_idx})
        else:
            status = 0

    return spans


def create_multiclass_label(pred, gold, tag_list, match_fun=None):
    """
    create multi-class label for span based tagging problem such as NER
    class label 0 means 'O', and class label i is tag_list[i-1]
    you can use the label for computing confusion matrix, computing f1, etc
    output: y_pred, y_true
    """
    assert len(pred) == len(gold), 'prediction and gold don\'t have the same length'
    if match_fun is None:
        match_fun = match_exact

    pairs = []
    for s in range(len(pred)):
        spans_p = create_span(pred[s], tag_list)
        spans_g = create_span(gold[s], tag_list)
        # for each predicted, compare with gold
        # remove matching element
        match_p = []
        match_g = []
        for i, span_p in enumerate(spans_p):
            for j, span_g in enumerate(spans_g):
                if match_fun(span_p, span_g):
                    match_p.append(i)
                    match_g.append(j)
                    pairs.append((span_p['tag_idx'], span_g['tag_idx'], s))
                    break

        spans_p_remaining = get_remaining(spans_p, match_p)
        spans_g_remaining = get_remaining(spans_g, match_g)

        # for each remaining pair it with O
        # O has id of -1
        for rem_p in spans_p_remaining:
            pairs.append((rem_p['tag_idx'], -1, s))

        for rem_g in spans_g_remaining:
            pairs.append((-1, rem_g['tag_idx'], s))

    # tag idx acts like an class id in typical multi-class classification
    y_pred, y_true, sent_idx = zip(*pairs)
    y_pred = np.array(y_pred, dtype=np.int64)
    y_true = np.array(y_true, dtype=np.int64)
    sent_idx = np.array(sent_idx, dtype=np.int64)
    # add by 1, because previously the label starts from -1 (other), 0 (tag 1), 1 (tag 2), ...
    y_pred += 1
    y_true += 1

    return y_pred, y_true, sent_idx

def create_vocab(sents, min_count=2, lowercase=True, replace_digits=True):
    freq = defaultdict(int)
    for sent in sents:
        for word in sent:
            if lowercase:
                word = word.lower()
            if replace_digits:
                word = re.sub(r'\d', '0', word)
            freq[word] += 1
    return {w for w, f in freq.items() if f >= min_count}
