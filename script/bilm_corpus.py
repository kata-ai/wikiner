import os
import re
# import sys
import pathlib
import argparse
import codecs
import glob
from collections import Counter, defaultdict

def word_tokenize(sentence, sep=r'(\W+)?'):
    return [x.strip() for x in re.split(sep, sentence) if x.strip()]

def print_txtfile(data, vocab, output_corpus):
    for line in data.readlines():
        sents = line.split(".\n")
        for sent in sents:
            # sent = line
            words = word_tokenize(sent)
            # update vocab using counter here
            words_count = Counter(words)
            for word, count in words_count.items():
                if word in vocab:
                    vocab[word] = vocab[word] + count
                else:
                    vocab[word] = count
            print(' '.join(words), file=output_corpus)
    return vocab

def print_conllfile(data, vocab, output_corpus, col_idx=0):
    tagged_words = []
    tagged_sents = [[]]
    lines = re.sub(r'(\n\s*)+\n', '\n\n', data.read())
    lines = lines.rstrip('\n').split('\n')
    for line in lines:
        line = line.rstrip('\n')
        if line:
            sent = line.split('\t')
            tagged_words.append(sent)
            tagged_sents[-1].append(tagged_words[-1])
        else:
            tagged_sents.append([])
    for sent in tagged_sents:
        words = [word_tag[col_idx] for word_tag in sent]
        # update vocab using counter here
        words_count = Counter(words)
        for word, count in words_count.items():
            if word in vocab:
                vocab[word] = vocab[word] + count
            else:
                vocab[word] = count
        print(' '.join(words), file=output_corpus)
    return vocab

def print_jsonlfile(data, vocab, output_corpus, key='content', sent_sep='.\n'):
    import json
    for line in data.readlines():
        json_line = json.loads(line)
        if key in json_line:
            content = json_line[key]
            sents = content.split(sent_sep)
            for i, sent in enumerate(sents):
                words = word_tokenize(sent)
                # update vocab using counter here
                words_count = Counter(words)
                for word, count in words_count.items():
                    if word in vocab:
                        vocab[word] = vocab[word] + count
                    else:
                        vocab[word] = count
                toline = ' '.join(words)
                if i != len(sents) - 1:
                    toline = toline + f' {sent_sep}'
                print(toline, file=output_corpus)
    return vocab

def main(args):
    print(args)
    # print(args.filepattern)
    files = glob.glob(args.filepattern)
    vocab = defaultdict()
    for corpus in files:
        if os.path.isfile(corpus):
            _, fname = os.path.split(corpus)
            output_file = os.path.join(args.output_dir, fname)
            mode = 'w+'
            print(output_file)
            if os.path.exists(output_file) and args.duplicate_append:
                print('file exists')
                mode = 'a'
            with open(output_file, mode=mode, encoding=args.encoding) as f:
                data = codecs.open(corpus, mode='r', encoding=args.encoding)
                if corpus.endswith('.txt'):
                    vocab = print_txtfile(data, vocab, f)
                elif corpus.endswith('.conll'):
                    vocab = print_conllfile(data, vocab, f, args.conll_idx)
                elif corpus.endswith('.jsonl'):
                    vocab = print_jsonlfile(data, vocab, f, args.json_key, args.sentence_separator)
                elif args.default_txt:
                    vocab = print_txtfile(data, vocab, f)
                elif args.default_jsonl:
                    vocab = print_jsonlfile(data, vocab, f, args.json_key, args.sentence_separator)
                elif args.default_conll:
                    vocab = print_conllfile(data, vocab, f, args.conll_idx)
                # p = pathlib.Path(output_file)
                # p.rename(p.with_suffix('.txt'))
    sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=args.reverse)
    print(len(sorted_vocab))
    # print(sorted_vocab)
    with open(args.vocab, mode='w', encoding=args.encoding) as vocab_file:
        for vocab in sorted_vocab[:args.max_vocab]:
            print(vocab[0], file=vocab_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""create vocab and corpus from recursive raw, conll or jsonl text files""")
    parser.add_argument('filepattern', metavar='FILE', help='path to the corpus file')
    parser.add_argument('vocab', metavar='FILE', help='filepath to the vocab file')
    parser.add_argument('-o', '--output-dir', metavar='DIR', default=os.getcwd(),
                        help='output directory (default: {})'.format(os.getcwd()))
    parser.add_argument('-i', '--conll-idx', metavar='IDX', type=int, default=0,
                        help='conll input word-idx')
    parser.add_argument('-k', '--json-key', metavar='KEY', type=str, default='content',
                        help='conll input word-idx')
    parser.add_argument('-ssep', '--sentence-separator', metavar='SSEP', type=str, default='. ',
                        help='sentence separator between lines if txt, string value, in jsonnl')
    parser.add_argument('-jsonl', '--default-jsonl', 
                        help='if extension not found use jsonl', action='store_true')
    parser.add_argument('-txt', '--default-txt', 
                        help='if extension not found use txt', action='store_true')
    parser.add_argument('-conll', '--default-conll', 
                        help='if extension not found use conll', action='store_true')
    parser.add_argument('-da', '--duplicate-append', 
                        help='if file exisit in output directory, just append the file', action='store_true')
    parser.add_argument('--encoding', default='utf-8',
                        help='file encoding (default: utf-8)')
    parser.add_argument('-m', '--max-vocab', metavar='vocab', type=int, default=100000,
                        help='maximum number of vocab keep for language model')
    parser.add_argument('-r', '--reverse', metavar='DESC', type=bool, default=True,
                        help='maximum number of vocab keep for language model')
    parser.add_argument('-t', '--test', metavar='test', type=float, default=0.1,
                        help='holdout percentage of corpus for validation')
    args = parser.parse_args()
    main(args)
