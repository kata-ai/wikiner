import os
import argparse
import itertools

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from helfer.corpus import CoNLLCorpus

from eval_utility import IOBTagger
from eval_utility import create_multiclass_label

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues'),
                          ylabel='True label',
                          xlabel='Predicted label'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot confusion matrix of a sequence labeling task with exact matching')
    parser.add_argument('reference', metavar='REF',
                        help='path to reference file in CoNLL format')
    parser.add_argument('hypothesis', metavar='HYP',
                        help='path to hypothesis file in CoNLL format')
    parser.add_argument('-o', '--save-to', required=True, help='where to save the plot')
    parser.add_argument('--encoding', default='utf-8',
                        help='file encoding (default: utf-8)')
    # parser.add_argument('--error-only', default=False, type=bool)
    parser.add_argument('--title', help='plot title')
    args = parser.parse_args()

    ref_root, ref_filename = os.path.split(args.reference)
    reference = CoNLLCorpus(ref_root, ref_filename, encoding=args.encoding)
    hyp_root, hyp_filename = os.path.split(args.hypothesis)
    hypothesis = CoNLLCorpus(hyp_root, hyp_filename, encoding=args.encoding)
    ref_tags = [[tag for _, tag in tagged_sent]
                for tagged_sent in reference.reader.tagged_sents()]
    hyp_tags = [[tag for _, tag in tagged_sent]
                for tagged_sent in hypothesis.reader.tagged_sents()]
    uniq_tags = set([IOBTagger.normalize_tag(tag) for tag in itertools.chain(*ref_tags)])
    uniq_tags.update([IOBTagger.normalize_tag(tag) for tag in itertools.chain(*hyp_tags)])
    # assert 'O' in uniq_tags
    # uniq_tags.remove('O')
    # uniq_tags.remove('V')
    uniq_tags = list(sorted(uniq_tags))
    # ref_tags, hyp_tags, _ = filter_srl_instance(ref_tags, hyp_tags)
    y_pred, y_test, _ = create_multiclass_label(hyp_tags, ref_tags, uniq_tags)
    labels = ['O'] + uniq_tags

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    # print(args.error_only)
    # if args.error_only:
    #     d3 = np.diag_indices_from(cnf_matrix)
    #     cnf_matrix[d3] = 0
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels,
                          title='Confusion matrix, without normalization')
    plt.savefig(args.save_to.replace('.jpg', '')+'-non-normalize'+'.jpg',
                dpi=300, bbox_inches='tight')

    # Plot normalized confusion matrix
    plt.figure()
    plt_title = args.title if args.title is not None else 'Normalized confusion matrix'
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True, title=plt_title)
    plt.savefig(args.save_to, dpi=300, bbox_inches='tight')
