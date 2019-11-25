import argparse

from ingredients.crf_utils import get_tagged_sents_and_words
from ingredients.crf_utils import sent2features
from ingredients.crf_utils import sent2stanfordfeats
from ingredients.crf_utils import sent2partial_labels
from ingredients.crf_utils import sent2stanford_partial

# def word2features(sent, idx, window_size):
#     word = sent[idx][0]
    
#     features = {
#         'bias': 1.0,
#         'word.lower()': word.lower(),
#         'char[-3..]': word[-3:],
#         'char[-2..]': word[-2:],
#         'char[..3]': word[:3],
#         'char[..2]': word[:2],
#         'word.isupper()': word.isupper(),
#         'word.istitle()': word.istitle(),
#         'word.isdigit()': word.isdigit(),
#         'contains_digit': any(char.isdigit() for char in word),
#     }
    
#     for j in range(window_size):
#         if idx-j-1 >= 0:
#             word = sent[idx-j-1][0]
#             features.update({
#                 str(idx-j-1) + ':word.lower()': word.lower(),
#                 str(idx-j-1) + ':char[-3..]': word[-3:],
#                 str(idx-j-1) + ':char[-2..]': word[-2:],
#                 str(idx-j-1) + ':char[..3]': word[:3],
#                 str(idx-j-1) + ':char[..2]': word[:2],
#                 str(idx-j-1) + ':word.istitle()': word.istitle(),
#                 str(idx-j-1) + ':word.isupper()': word.isupper(),
#                 str(idx-j-1) + ':word.isdigit()': word.isdigit(),
#                 str(idx-j-1) + ':contains_digit': any(char.isdigit() for char in word),

#             })
#         if idx+j+1 < len(sent):
#             word = sent[idx+j+1][0]
#             features.update({
#                 str(idx+j+1) + ':word.lower()': word.lower(),
#                 str(idx+j+1) + ':word[-3..]': word[-3:],
#                 str(idx+j+1) + ':word[-2..]': word[-2:],
#                 str(idx+j+1) + ':word[..3]': word[:3],
#                 str(idx+j+1) + ':word[..2]': word[:2],
#                 str(idx+j+1) + ':word.istitle()': word.istitle(),
#                 str(idx+j+1) + ':word.isupper()': word.isupper(),
#                 str(idx+j+1) + ':word.isdigit()': word.isdigit(),
#                 str(idx+j+1) + ':contains_digit': any(char.isdigit() for char in word),
#             })
    
#     if idx == 0:
#         features['BOS'] = True

#     if idx == len(sent)-1:
#         features['EOS'] = True

#     return features

# def sent2features(sent, window_size):
#     return [word2features(sent, i, window_size) for i in range(len(sent))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract features for CRFSuite input format')
    parser.add_argument('file', help='CoNLL file from which the features will be extracted')
    parser.add_argument('--window-size', default=0, type=int, help='window sizes')
    parser.add_argument('--features', default='default', choices=['default', 'stanford'], \
                        type=str, help='features prep, default, or stanford-ner')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    args = parser.parse_args()

    pre = ['B', 'I']
    ent = ['Person', 'Place', 'Organisation']

    labels = [f'{x}-{y}' for x in pre for y in ent] + ['O']
    # print(labels)

    corpus, _ = get_tagged_sents_and_words(args.file)
    for tagged_sent in corpus:
        if args.features == 'default':
            feats = sent2features(tagged_sent, args.window_size)
            tags = sent2partial_labels(tagged_sent, labels=labels)
        elif args.features == 'stanford':
            feats = sent2stanfordfeats(tagged_sent)
            tags = sent2stanford_partial(tagged_sent, labels=labels)
        for tag, feature in zip(tags, feats):
            # feature = ['{}={}'.format(k, v) for k, v in feature.items()]
            feat = []
            for k, v in feature.items():
                if k.split(':')[0].isdigit():
                    weight_name = k.split(':')
                    feat.append('{}={}:{}'.format(''.join(weight_name[1:]).strip(':'), v, weight_name[0]))
                else:
                    feat.append('{}={}'.format(k.strip(':'), v))
            feature = feat
            print('{}\t{}'.format(tag, '\t'.join(feature)))
        print()
