import logging
import re
import io
import os
from os.path import splitext
import json
import argparse

from nltk import sent_tokenize, word_tokenize
from gensim.corpora import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedDocument


def tokenize(sentence, sep=r'(\W+)?'):
    return [x.lower().strip() for x in re.split(sep, sentence)]


def check_model_dir(model_snapshot_dir):
    """
    Check whether model dir already exists
    """
    if os.path.isdir(model_snapshot_dir):
        logging.info("snapshot dir exists")
        return True
    else:
        return False


def create_model_dir(model_dir):
    """
    Create directory for model bin
    """
    try:
        if check_model_dir(model_dir):
            return model_dir
        os.makedirs(model_dir)
        return model_dir
    except OSError as err:
        logging.warning(err)
        return model_dir


class LineSentences(object):
    def __init__(self, dirname, wikipath=None, lower=True):
        self.dirname = dirname
        self.wiki = None
        if wikipath:
            self.wiki = WikiCorpus(wikipath, lemmatize=False, dictionary={}, lower=lower)
            self.wiki.metadata = False

    def __iter__(self):
        # if self.wiki:
        for content in self.wiki.get_texts():
            # print(content)
            yield content
        for fname in os.listdir(self.dirname):
            _, ext = splitext(fname)
            if ".txt" in ext:
                for line in open(os.path.join(self.dirname, fname)):
                    line = line.rstrip('\n')
                    words = word_tokenize(line)
                    if words:
                        # print(words)
                        yield words
        


class TaggedWikiDocument(object):
    def __init__(self, wiki, with_title=True):
        self.wiki = wiki
        self.wiki.metadata = True
        self.with_title = with_title

    def __iter__(self):
        for content, (_, title) in self.wiki.get_texts():
            if self.with_title:
                yield TaggedDocument([c.decode("utf-8") for c in content], [title])
            else:
                yield tokenize(content)


def train_word2vec_model(train_dir, model_dir, model_name, train_wikidump=None, lower=True, **model_params):
    """Train word2vec. if sg==0, use cbow, else skip-gram.
    `training_messages` is a 2D list or a SentenceGenerator"""
    sentences = LineSentences(train_dir, wikipath=train_wikidump, lower=lower)
    model = Word2Vec(sentences, **model_params)
    model.init_sims(replace=True)

    save_embedding_model(model_dir, model, model_name)
    return model


def save_embedding_model(model_dir, embedding_model, model_name):
    """Save embedding"""
    model_dir = create_model_dir(model_dir)
    save_dir = os.path.join(model_dir, model_name)
    # embedding_model.save(save_dir)
    embedding_model.wv.save_word2vec_format(save_dir + '.txt.gz')
    embedding_model.wv.save_word2vec_format(save_dir + '.txt')


def train_word2vec_from_args(args):
    kwargs = vars(args)
    model = train_word2vec_model(**kwargs)
    try:
        while True:
            sim_word = input("sim word : > ")
            topn_word = int(input("topn : > "))
            print('most similar word')
            print(model.most_similar(sim_word, None, int(topn_word)))
            pos_word = input("pos word : > ")
            neg_word = input("neg word : > ")
            topn_word = int(input("topn : > "))
            # print(model[test_word])
            # print(model.similar_by_word(test_word, topn_word))
            print('most similar')
            print(model.most_similar(pos_word, neg_word, int(topn_word)))
            a_word = input("a word : > ")
            b_word = input("b word : > ")
            c_word = input("neg word : > ")
            topn_word = int(input("topn : > "))
            print('most similar cosmul')
            print(model.most_similar_cosmul([a_word, b_word], [c_word], int(topn_word)))
    except KeyboardInterrupt as e:
        print(e)
        exit(0)
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='pretrain word embedding model using gensim word2vec'
    )
    parser.add_argument('train_dir', type=str, help='train data dirpath contain txt files')
    parser.add_argument('model_dir', type=str, help='model save dirpath in serialized format')
    parser.add_argument('model_name', type=str, help='model filename to saved')
    parser.add_argument('--train-wikidump', type=str, help='train wikidump using absolute/relative filepath')
    parser.add_argument('--size', type=int, help='words vector dimension size', default=300)
    parser.add_argument('--min_count', type=int, help='min word count to added to vocab', default=1)
    parser.add_argument('--alpha', type=float, help='learning rate', default=0.25)
    parser.add_argument('--max_vocab_size', type=int, help='max vocabulary size allocated', default=1000000)
    parser.add_argument('--window', type=int, help='context window size of n-gram', default=5)
    parser.add_argument('--sample', type=float, help='sample', default=0.25)
    parser.add_argument('--min_alpha', type=float, help='minimum value of decreased alpha', default=0.0001)
    parser.add_argument('--sg', type=int, help='use skipgram model option 1 or 0', default=1)
    parser.add_argument('--hs', type=int, help='use hierarchical sampling model option 1 or 0', default=0)
    parser.add_argument('--cbow_mean', type=int, help='cbow_mean', default=1)
    parser.add_argument('--lower', type=bool, help='lowercase token or not', default=True)
    parser.add_argument('--iter', type=int, help='number of iteration performed', default=5)
    train_args = parser.parse_args()
    train_word2vec_from_args(train_args)
