import re
import string

import numpy as np

# from helfer.corpus import CoNLLCorpus
from helfer.evaluation import evaluate_conll

def get_tagged_sents_and_words(*args):
    tagged_words = []
    tagged_sents = [[]]
    for file in args:
        dataset = open(file,'r')
        # remove multiple lines in dataset
        lines = re.sub(r'(\n\s*)+\n', '\n\n', dataset.read())
        lines = lines.rstrip('\n').split('\n')
        for line in lines:
            line = line.rstrip('\n')
            if line:
                sent = line.split('\t')
                tagged_words.append(sent)
                tagged_sents[-1].append(tagged_words[-1])
            else:
                tagged_sents.append([])
    # remove empty list
    # tagged_sents = [x for x in tagged_sents if x != []]
    return np.array(tagged_sents), np.array(tagged_words)


def wordshape(text):
    t1 = re.sub('[A-Z]', 'X',text)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)

def get_pattern(s):
    pattern = []
    for c in s:
        if c in string.ascii_lowercase:
            pattern.append('a')
        elif c in string.ascii_uppercase:
            pattern.append('A')
        elif c in string.digits:
            pattern.append('0')
        else:
            pattern.append('-')
    return ''.join(pattern)


def shorten_pattern(s):
    t = 0
    res = []
    while t < len(s):
        res.append(s[t])
        while t < len(s) and s[t] == res[-1]:
            t += 1
    return ''.join(res)

def word2features(sent, idx, window_size):
    word = sent[idx][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'contains_digit': any(char.isdigit() for char in word),
    }
    
    for j in range(window_size):
        if idx-j-1 >= 0:
            word = sent[idx-j-1][0]
            features.update({
                str(idx-j-1) + ':word.lower()': word.lower(),
                str(idx-j-1) + ':word[-3:]': word[-3:],
                str(idx-j-1) + ':word[-2:]': word[-2:],
                str(idx-j-1) + ':word[:3]': word[:3],
                str(idx-j-1) + ':word[:2]': word[:2],
                str(idx-j-1) + ':word.istitle()': word.istitle(),
                str(idx-j-1) + ':word.isupper()': word.isupper(),
                str(idx-j-1) + ':word.isdigit()': word.isdigit(),
                str(idx-j-1) + ':contains_digit': any(char.isdigit() for char in word),

            })
        if idx+j+1 < len(sent):
            word = sent[idx+j+1][0]
            features.update({
                str(idx+j+1) + ':word.lower()': word.lower(),
                str(idx+j+1) + ':word[-3:]': word[-3:],
                str(idx+j+1) + ':word[-2:]': word[-2:],
                str(idx+j+1) + ':word[:3]': word[:3],
                str(idx+j+1) + ':word[:2]': word[:2],
                str(idx+j+1) + ':word.istitle()': word.istitle(),
                str(idx+j+1) + ':word.isupper()': word.isupper(),
                str(idx+j+1) + ':word.isdigit()': word.isdigit(),
                str(idx+j+1) + ':contains_digit': any(char.isdigit() for char in word),
            })
    
    if idx == 0:
        features['BOS'] = True

    if idx == len(sent)-1:
        features['EOS'] = True

    return features

def word2lessfeatures(sent, idx, window_size):
    word = sent[idx][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'contains_digit': any(char.isdigit() for char in word),
        'word.shape': wordshape(word),
    }
    
    for j in range(window_size):
        if idx-j-1 >= 0:
            word = sent[idx-j-1][0]
            features.update({
                str(j-1) + ':word.lower()': word.lower(),
                # str(j-1) + ':word[-3:]': word[-3:],
                # str(j-1) + ':word[-2:]': word[-2:],
                # str(j-1) + ':word[:3]': word[:3],
                # str(j-1) + ':word[:2]': word[:2],
                # str(j-1) + ':word.istitle()': word.istitle(),
                # str(j-1) + ':word.isupper()': word.isupper(),
                # str(j-1) + ':word.isdigit()': word.isdigit(),
                str(j-1) + ':contains_digit': any(char.isdigit() for char in word),

            })
            if idx-j-2 >= 0:
                wordprev = sent[idx-j-2][0]
                features.update({
                    f'{str(j-2)}-{str(j-1)}:word.lower()': f'{word.lower()}-{wordprev.lower()}',
                })

        if idx+j+1 < len(sent):
            word = sent[idx+j+1][0]
            features.update({
                str(j+1) + ':word.lower()': word.lower(),
                # str(j+1) + ':word[-3:]': word[-3:],
                # str(j+1) + ':word[-2:]': word[-2:],
                # str(j+1) + ':word[:3]': word[:3],
                # str(j+1) + ':word[:2]': word[:2],
                # str(j+1) + ':word.istitle()': word.istitle(),
                # str(j+1) + ':word.isupper()': word.isupper(),
                # str(j+1) + ':word.isdigit()': word.isdigit(),
                # str(j+1) + ':contains_digit': any(char.isdigit() for char in word),
            })
            if idx+j+2 < len(sent):
                wordnext = sent[idx+j+2][0]
                features.update({
                    f'{str(j-2)}-{str(j-1)}:word.lower()': f'{word.lower()}-{wordnext.lower()}',
                })
    
    if idx == 0:
        features['BOS'] = True

    if idx == len(sent)-1:
        features['EOS'] = True

    return features

def stanford_crffeat(sent, idx):
    """
    CRF suite features using output from Stanford NER Feature Extractor
    see https://github.com/stanfordnlp/CoreNLP/blob/master/src
        edu.stanford.nlp.ie.crf.CRFFeatureExporter

    CRFFeatureExporter \
        -prop crfClassifierPropFile \
        -trainFile inputFile \
        -exportFeatures outputFile
    """
    instance_feature = {}
    for feature in sent[idx][2:]:
        template = feature.split('-')
        instance_feature[template[-1]] = '-'.join(template[:-1])
    return instance_feature


def sent2stanfordfeats(sent):
    return [stanford_crffeat(sent, i) for i in range(len(sent))]

def sent2stanfordlabels(sent, col=1):
    return [word_tag_feat[col].upper() for word_tag_feat in sent]

def sent2features(sent, window_size):
    return [word2features(sent, i, window_size) for i in range(len(sent))]

def sent2lessfeatures(sent, window_size):
    return [word2lessfeatures(sent, i, window_size) for i in range(len(sent))]

def sent2labels(sent):
    return [label.upper() for token, label in sent]

def sent2labels_colmap(sent, col=0):
    return [word_tag[col].upper() for word_tag in sent]

def sent2partial_labels(sent, labels, label_unk = 'O', ):
    label_sent = []
    for idx, (word, label) in enumerate(sent):
        # simple istitle and isupper heuristic guided for partial-annotation wikiner
        if label == label_unk and (word.istitle() or word.isupper()) and not idx == 0:
            label = '|'.join(labels)
        # TODO add if and else if the word is present in kb or entity dictionary
        # TODO add if and else if the word is present is noun or non-entity pos tags
        # TODO add if and else constrain based dictionary mapping based on previous word
        label_sent.append(label.upper())
    return label_sent

def sent2stanford_partial(sent, labels, word_col=0, tag_col=1, label_unk='O'):
    label_sent = []
    for idx, word_tag_feat in enumerate(sent):
        label = word_tag_feat[tag_col]
        word = word_tag_feat[word_col]
        # simple istitle and isupper heuristic guided for partial-annotation wikiner
        if label == label_unk and (word.istitle() or word.isupper()) and not idx == 0:
            label = '|'.join(labels)
        # TODO add if and else if the word is present in kb or entity dictionary
        # TODO add if and else if the word is present is noun or non-entity pos tags
        # TODO add if and else constrain based dictionary mapping based on previous word
        label_sent.append(label.upper())
    return label_sent

def sent2tokens(sent):
    return [token for token, label in sent]

def evaluate(reference, hypothesis):
    evaluable = [zip(*sent_pair) for sent_pair in zip(reference, hypothesis)]
    return evaluate_conll(evaluable)
