import os
import re
from argparse import ArgumentParser
from collections import Counter, defaultdict

from helfer.corpus import CoNLLCorpus


def transform_word(word, lowercase=True, replace_digits=True):
    if lowercase:
        word = word.lower()
    if replace_digits:
        word = re.sub(r'\d', '0', word)
    return word


def vocab_words(tagged_corpus: CoNLLCorpus,
                has_tag: str='V',
                lowercase: bool=True,
                replace_digits: bool=True):
    words = []
    for tagged_sent in tagged_corpus.reader.tagged_sents():
        if any(has_tag in t[1] for t in tagged_sent):
            for t in tagged_sent:
                word = transform_word(t[0], lowercase, replace_digits)
                words.append(word)
    return words


DEFAULT_MIN_WORD_COUNT = 2
DEFAULT_ENCODING = 'utf-8'

parser = ArgumentParser(
    description='Compute OOV rate from the given training and test CoNLL corpus')
parser.add_argument('train_file', metavar='TRAIN',
                    help='path to train file in CoNLL format')
parser.add_argument('test_file', metavar='TEST',
                    help='path to test file in CoNLL format')
parser.add_argument('--encoding', default=DEFAULT_ENCODING,
                    help='file encoding (default: {})'.format(DEFAULT_ENCODING))
parser.add_argument('--min-word-count', '-c', metavar='COUNT', type=int,
                    default=DEFAULT_MIN_WORD_COUNT,
                    help='min count for a word to be included in the vocabulary '
                    '(default: {})'.format(DEFAULT_MIN_WORD_COUNT))
parser.add_argument('--no-lowecase', action='store_false', dest='lowercase',
                    help='do not lowercase words')
parser.add_argument('--no-replace-digits', action='store_false', dest='replace_digits',
                    help='do not replace digits')
parser.add_argument('--per-tag', action='store_true', help='compute per-tag OOV rate')
args = parser.parse_args()

train_dir, train_filename = os.path.split(args.train_file)
test_dir, test_filename = os.path.split(args.test_file)
train_corpus = CoNLLCorpus(train_dir, train_filename, encoding=args.encoding)
test_corpus = CoNLLCorpus(test_dir, test_filename, encoding=args.encoding)

train_words = [transform_word(word, lowercase=args.lowercase,
                              replace_digits=args.replace_digits)
               for word in train_corpus.reader.words()]
# train_words = vocab_words(train_corpus, lowercase=args.lowercase,
#                           replace_digits=args.replace_digits)

word_counter = Counter(train_words)
vocab = {word for word, count in word_counter.items() if count >= args.min_word_count}

vocab_tag = defaultdict(set)
for word, tag in train_corpus.reader.tagged_words():
    word = transform_word(word, lowercase=args.lowercase,
                          replace_digits=args.replace_digits)
    tagname = tag if tag == 'O' else tag[2:]
    vocab_tag[tagname].add(word)

if not args.per_tag:
    test_words = [transform_word(word, lowercase=args.lowercase,
                                 replace_digits=args.replace_digits)
                  for word in test_corpus.reader.words()]
    # test_words = vocab_words(test_corpus, lowercase=args.lowercase,
    #                          replace_digits=args.replace_digits)
    # print(test_words)
    unique_test_words = set(test_words)
    oov_words = unique_test_words - vocab
    print('OOV rate is {:.2%}'.format(len(oov_words)/len(unique_test_words)))
else:
    unique_test_words = defaultdict(set)
    for word, tag in test_corpus.reader.tagged_words():
        word = transform_word(word, lowercase=args.lowercase,
                              replace_digits=args.replace_digits)
        tagname = tag if tag == 'O' else tag[2:]
        unique_test_words[tagname].add(word)
    # for tagged_test in test_corpus.reader.tagged_sents():
    #     if any('V' in t for _, t in tagged_test):
    #         for word, tag in tagged_test:
    #             trans_word = transform_word(word, args.lowercase,
    #                                         args.replace_digits)
    #             tagname = tag if tag == 'O' else tag[2:]
    #             # if 'V' not in tagname:
    #             unique_test_words[tagname].add(word)
    oov_words = {tagname: unique_test_words[tagname] - vocab for tagname in unique_test_words}
    overlap_words = {tagname: unique_test_words[tagname].intersection(vocab) for tagname in unique_test_words}
    overlap_words_and_tags = {tagname: unique_test_words[tagname].intersection(vocab_tag[tagname]) for tagname in unique_test_words}
    for tagname in oov_words:
        # print('OOV rate {} is {:.2%}'.format(
        #     tagname, len(oov_words[tagname])/len(unique_test_words[tagname])))
        # print('Overlap rate {} is {:.2%}'.format(
        #     tagname, len(overlap_words[tagname])/len(unique_test_words[tagname])))
        oov_rate = len(oov_words[tagname])/len(unique_test_words[tagname])
        overlap_rate = len(overlap_words[tagname])/len(unique_test_words[tagname])
        overlap_tp_rate = len(overlap_words_and_tags[tagname])/len(unique_test_words[tagname])
        print('OOV rate {} is {:.2%}, Overlap rate is {:.2%},  Overlap TP rate is {:.2%}'.format(
              tagname, oov_rate, overlap_rate, overlap_tp_rate))
        print(overlap_words[tagname])
