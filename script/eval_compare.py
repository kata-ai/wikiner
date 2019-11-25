import os
from collections import namedtuple
import argparse
import itertools
import random
import re

import numpy as np
from helfer.corpus import CoNLLCorpus
from eval_utility import IOBTagger
from eval_utility import create_multiclass_label
from eval_utility import create_vocab


Element = namedtuple('Element', 'predicted true sent_idx')

def examine_sample(multiclass_labels, true_label, predicted_label, label_pred, label_gold,
                   true_texts, vocab, num_samples=1):
    sentences = [elem.sent_idx for elem in multiclass_labels
                 if elem.predicted == predicted_label and elem.true == true_label]

    for random_sent in random.sample(sentences, k=num_samples):
        print()
        sent = true_texts[random_sent]
        x_word = [word if re.sub(r'\d', '0', word.lower()) in vocab else '<unk>'
                  for word in sent]
        y_pred = label_pred[random_sent]
        y_true = label_gold[random_sent]
        lengths = np.array([[len(w) for w in sent],
                            [len(w) for w in x_word],
                            [len(w) for w in y_pred],
                            [len(w) for w in y_true]])
        max_len = np.max(lengths, axis=1)
        max_len += 2
        template = '{word:{lengths[0]}}{x:{lengths[1]}}{yt:{lengths[2]}}{yp:{lengths[3]}}'
        print(template.format(lengths=max_len, word='true', x='vocab', yt='gold', yp='pred'))
        for word, x, yt, yp in zip(sent, x_word, y_true, y_pred):
            template = '{word:{lengths[0]}}{x:{lengths[1]}}{yt:{lengths[2]}}{yp:{lengths[3]}}'
            print(template.format(lengths=max_len, word=word, x=x, yt=yt, yp=yp))

def compare_sample_hyp(multiclass_labels,
                       true_label, predicted_a, predicted_b,
                       label_gold, label_pred_a, label_pred_b,
                       true_texts, vocab, num_samples=1):
    sentences = [elem.sent_idx for elem in multiclass_labels
                 if elem.predicted == predicted_a
                 and elem.true == true_label
                 and elem.predicted]

    for random_sent in random.sample(sentences, k=num_samples):
        sent = true_texts[random_sent]
        x_word = [word if re.sub(r'\d', '0', word.lower()) in vocab else '<unk>'
                  for word in sent]
        y_pred_a = label_pred_a[random_sent]
        y_true = label_gold[random_sent]
        lengths = np.array([[len(w) for w in sent],
                            [len(w) for w in x_word],
                            [len(w) for w in y_pred_a],
                            [len(w) for w in y_true]])
        max_len = np.max(lengths, axis=1)
        max_len += 2
        template = '{word:{lengths[0]}}{x:{lengths[1]}}{yt:{lengths[2]}}{yp:{lengths[3]}}'
        print(template.format(lengths=max_len, word='true', x='vocab', yt='gold', yp='pred'))
        for word, x, yt, yp in zip(sent, x_word, y_true, y_pred_a):
            template = '{word:{lengths[0]}}{x:{lengths[1]}}{yt:{lengths[2]}}{yp:{lengths[3]}}'
            print(template.format(lengths=max_len, word=word, x=x, yt=yt, yp=yp))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample sentence for error analysis')
    parser.add_argument('reference', metavar='REF', help='reference file in CoNLL format')
    parser.add_argument('hypothesis', metavar='HYP', help='hypothesis file in CoNLL format')
    parser.add_argument('-t', '--train-file', metavar='FILE', required=True,
                        help='path to training CoNLL file to build vocab')
    parser.add_argument('--ref-tag', metavar='TAG', required=True, help='reference tag '
                        'without B-/I- prefix (required)')
    parser.add_argument('--hyp-tag', metavar='TAG', help='hypothesis tag without B-/I- '
                        'prefix (default: same as --ref-tag)')
    parser.add_argument('--encoding', default='utf-8',
                        help='file encoding (default: utf-8)')
    parser.add_argument('-k', '--num-samples', type=int, default=1,
                        help='how many samples to draw')
    args = parser.parse_args()

    ref_root, ref_filename = os.path.split(args.reference)
    reference = CoNLLCorpus(ref_root, ref_filename, encoding=args.encoding)
    hyp_root, hyp_filename = os.path.split(args.hypothesis)
    hypothesis = CoNLLCorpus(hyp_root, hyp_filename, encoding=args.encoding)
    out_root, out_filename = os.path.split(args.train_file)
    train_corpus = CoNLLCorpus(out_root, out_filename, encoding=args.encoding)
    ref_texts = [[word for word, _ in tagged_sent]
                 for tagged_sent in reference.reader.tagged_sents()]
    ref_tags = [[tag for _, tag in tagged_sent]
                for tagged_sent in reference.reader.tagged_sents()]
    hyp_tags = [[tag for _, tag in tagged_sent]
                for tagged_sent in hypothesis.reader.tagged_sents()]
    train_texts = [[word for word, _ in tagged_sent]
                   for tagged_sent in train_corpus.reader.tagged_sents()]
    uniq_tags = set([IOBTagger.normalize_tag(tag) for tag in itertools.chain(*ref_tags)])
    uniq_tags.update([IOBTagger.normalize_tag(tag) for tag in itertools.chain(*hyp_tags)])
    assert 'O' in uniq_tags
    uniq_tags.remove('O')
    uniq_tags = list(sorted(uniq_tags))
    id2label = ['O'] + uniq_tags
    vocab = create_vocab(train_texts)

    out = create_multiclass_label(hyp_tags, ref_tags, uniq_tags)
    multiclass_labels = [Element(id2label[pred], id2label[true], idx)
                         for pred, true, idx in zip(*out)]
    examine_sample(
        multiclass_labels, args.ref_tag,
        args.hyp_tag if args.hyp_tag is not None else args.ref_tag, hyp_tags, ref_tags,
        ref_texts, vocab, num_samples=args.num_samples)
    # compare_sample_hyp(multiclass_labels, args.ref_tag,
    #                    args.hyp_tag if args.hyp_tag is not None else args.ref_tag,
    #                    hyp_tags, ref_tags)
