from __future__ import print_function, unicode_literals

import argparse
import codecs
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

from six import with_metaclass  # Python 2 and 3 compatibility

from helfer.corpus import CoNLLCorpus, split


class KFoldSplit:
    def __init__(self, parser):
        self._parser = parser

    @property
    def parser(self):
        return self._parser

    def __call__(self, args):
        corpus = CoNLLCorpus(*os.path.split(args.corpus_file), encoding=args.encoding)
        all_unique_tags = {self.get_unique_tags(s) for s in corpus.reader.tagged_sents()}
        unique_tags_idx = {tag: idx for idx, tag in enumerate(all_unique_tags)}
        print(unique_tags_idx)

        kfold = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        tags_seq = []
        tags_str = []
        groups = []
        only_o = set('O')
        non_o_tag = 0
        for tagged_sent in corpus.reader.tagged_sents():
            # tagset = set([tag for _, tag in tagged_sent])
            tagset = self.get_unique_tags(tagged_sent)
            if not(tagset == only_o):
                non_o_tag += 1
                # sent.append([tok for tok, _ in tagged_sent])
                # tags.append([tag for _, tag in tagged_sent])
                tags_seq.append([(tok, tag) for tok, tag in tagged_sent])
                tags = [tag for _, tag in tagged_sent]
                tags_str.append(' '.join(tags))
                groups.append(unique_tags_idx[tagset])
        # print(f'non_o_tag : {non_o_tag}')
        # print(groups)
        sent_vect = np.asanyarray(tags_seq)

        tag_vectorizer = TfidfVectorizer(encoding=args.encoding)
        tags_vect = tag_vectorizer.fit_transform(tags_str)
        for fid, (trainidx, testidx) in enumerate(kfold.split(tags_vect)):
            trainset = sent_vect[trainidx]
            testset = sent_vect[testidx]

            print(fid)
            print('train length: {}'.format(len(trainset)))
            print('test length: {}'.format(len(testset)))
            # print(fid)
            outputs = [(trainset, 'train'), (testset, 'test')]
            for output_set, name in outputs:
                # Use codecs.open for Python 2 compat
                with codecs.open(os.path.join(args.output_dir, f'{name}-{fid}.conll'), 'w',
                                 encoding=args.encoding) as f:
                    CoNLLCorpus.write_tagged_sents(output_set, file=f)

            devset = []
            if args.active_learn:
                trainls = trainset.tolist()
                labeled, unlabeled, rest = split(trainls,
                                                 dev_prob=args.unlabeled_prob-0.01,
                                                 test_prob=0.01,
                                                 group_by=self.get_unique_tags,
                                                 seed=args.seed)
                print('labeled length: {}'.format(len(labeled)))
                print('unlabeled length: {}'.format(len(unlabeled)))
                unlabeled.extend(rest)
                devset = unlabeled
                assert len(labeled)+len(unlabeled) == len(trainset)
                outputs = [(labeled, 'labeled'), (unlabeled, 'unlabeled')]
                for output_set, name in outputs:
                    # Use codecs.open for Python 2 compat
                    with codecs.open(os.path.join(args.output_dir, f'{name}-{fid}.conll'), 'w',
                                     encoding=args.encoding) as f:
                        CoNLLCorpus.write_tagged_sents(output_set, file=f)

            if args.verbose:
                for unique_tags in sorted(all_unique_tags):
                    print('tags', ' '.join(unique_tags), end=' | ', file=sys.stderr)
                    n_total = 0
                    for output_set, name in outputs:
                        n_sents = sum(self.get_unique_tags(s) == unique_tags for s in output_set)
                        n_total += n_sents
                        print(name, n_sents, end=' | ', file=sys.stderr)
                    print('total', n_total, file=sys.stderr)
                if args.active_learn:
                    print('TOTAL | train', len(trainset), '| dev', len(devset), '| test', len(testset),
                          '| total', len(trainset) + len(devset) + len(testset), file=sys.stderr)

    @classmethod
    def get_unique_tags(cls, tagged_sent):
        _, tags = zip(*tagged_sent)
        unique_tags = {cls.remove_prefix(tag) for tag in tags}
        return tuple(sorted(cls.remove_other_tag(unique_tags)))

    @staticmethod
    def remove_prefix(tag):
        return tag[2:] if tag.startswith('B-') or tag.startswith('I-') else tag

    @staticmethod
    def remove_other_tag(unique_tags):
        try:
            unique_tags.remove('O')
        except KeyError:
            pass  # we don't care
        if not unique_tags:
            unique_tags.add('O')
        return unique_tags


def main():
    parser = argparse.ArgumentParser(
        description="""KFold command line interface.
        Split sentences in a CoNLL corpus with IOB tagging into several k-fold set

        The split will be done via stratified sampling based on each sentence's set of
        unique tags (ignoring O tags). The sets will be saved to train.conll, dev.conll,
        and test.conll files in the output directory respectively.""")
    parser.add_argument('corpus_file', metavar='FILE', help='path to the corpus file')
    parser.add_argument('-o', '--output-dir', metavar='DIR', default=os.getcwd(),
                        help='output directory (default: {})'.format(os.getcwd()))
    parser.add_argument('--encoding', default='utf-8',
                        help='file encoding (default: utf-8)')
    parser.add_argument('-f', '--fold', metavar='FOLD', type=int, default=5,
                        help='number of k-fold spit return')
    parser.add_argument('-al', '--active-learn', action='store_true',
                        help='')
    # parser.add_argument('-d', '--dev-prob', metavar='PROB', type=float, default=0.2,
    #                     help='proportion of dev from train in cross validation (default: 0.1)')
    parser.add_argument('-u', '--unlabeled-prob', metavar='PROB', type=float, default=0.5,
                        help='proportion of test set (default: 0.1)')
    parser.add_argument('--seed', help='random generator seed')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print splitting statistics to standard error')
    # args.command_fn(args)
    ksplit = KFoldSplit(parser)
    args = parser.parse_args()
    ksplit(args=args)


if __name__ == '__main__':
    main()
