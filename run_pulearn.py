import os
import logging
from typing import List

import yaml
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

# from pymongo import MongoClient
# from sacred.observers import MongoObserver
from sacred import Experiment
from sacred.run import Run
from pywsl.cpe.cpe_ene import alpha_dist
from pywsl.utils.comcalc import bin_clf_err

from ingredients.pu_classifier import EntityDetectionPU
from ingredients.crf_utils import get_tagged_sents_and_words
from ingredients.crf_utils import sent2labels_colmap

from ingredients.pu_utils import sent2pufeatures as sent2features
# from ingredients.crf_utils import sent2features
# from ingredients.crf_utils import sent2lessfeatures as sent2features
from ingredients.gen_configs import generate_params

SECRET = os.environ.get('SACRED_KEY', None)
MONGOL = f'mongodb://fariz:{SECRET}@ml-tools.kata.net:27017/sacredFariz'

ex = Experiment('run_pulearning')

# client = MongoClient(MONGOL)

# ex.observers.append(MongoObserver.create(
#     url=MONGOL, db_name='sacredFariz'))

# db = client['sacredFariz']
# runs = db['runs']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
logger.addHandler(handler)
ex.logger = logger


@ex.config
def default_config():
    train_files = ['train.conll']
    dev_files = ['dev.conll']
    test_files = ['test.conll']
    dirpath = '/home/fariz/workspace/dbpedia/dee'
    num_experiments = 30
    num_window_sizes = 4
    retry_limit = 10
    pos_label = ['B-PERSON', 'B-PLACE', 'B-ORGANISATION',
                 'I-PERSON', 'I-PLACE', 'I-ORGANISATION']

@ex.named_config
def cross_val_config():
    train_files = [f'train-{i}.conll' for i in range(5)]
    dev_files = [f'test-{i}.conll' for i in range(5)]
    dirpath = '/home/fariz/workspace/dbpedia/dee'
    num_experiments = 30
    num_window_sizes = 5
    retry_limit = 10


@ex.capture
def process_data(corpus, window_size, label_idx=1, 
                 pos_label=None, unlabeled=True, 
                 no_label=False, _run=None, _log=None):
    sents, _ = get_tagged_sents_and_words(corpus)

    X = [sent2features(s, window_size) for s in sents]
    y = [sent2labels_colmap(s, label_idx) for s in sents]

    # flatten the data
    X_flat = []
    y_flat = []
    label_flat = []
    word_flat = []

    for sent_raw, sent_feat, sent_tag in zip(sents, X, y):
        for word, feat, tag in zip(sent_raw, sent_feat, sent_tag):
            X_flat.append(feat)
            word_flat.append(word)
            label_flat.append(tag)
            if tag in pos_label and not(no_label):
                y_flat.append(1.)
            elif unlabeled:
                y_flat.append(0.)
            else:
                y_flat.append(-1.)

    X_flat = np.asanyarray(X_flat)
    y_flat = np.asanyarray(y_flat)
    return X_flat, y_flat, label_flat

@ex.capture
def evaluate_pu(y_gold, y_pred, labels, prior=0.5, _run: Run = None, _log: logger = None):
    p, r, f, _ = precision_recall_fscore_support(y_gold, y_pred, labels=labels)
    _log.info(f'label : {labels}')
    _run.info[f'overall_f1'] = f
    _log.info(f'overall_f1: {list(zip(labels, f))}')
    _run.info[f'overall_precision'] = p
    _log.info(f'overall_precision: {list(zip(labels, p))}')
    _run.info[f'overall_recall'] = r
    _log.info(f'overall_recall: {list(zip(labels, r))}')

    err = bin_clf_err(y_pred, y_gold, prior)
    _log.info(f'bin_clf_err: {err}')
    _run.info['bin_clf_err'] = err
    return p, r, f, err

def sample_output(data, y_gold, y_pred, labels, _run, _log):
    # true positive y_gold=1, y_pred=1
    labels = np.asanyarray(labels)
    _log.info(f'total data : {len(labels)}')

    pos_label = y_pred==1
    _log.info(f'total positive instance : {len([p for p in pos_label if p])}')

    tp_sample = [p==1 and g==1 for p, g in zip(y_pred, y_gold)]
    tp_ = list(zip(data[tp_sample], labels[tp_sample], y_gold[tp_sample], y_pred[tp_sample]))
    _log.info(f'total true positive : {len(tp_)}')

    for _, t, g, p in tp_[:5]:
        _log.info(f'{t}\t{g}\t{p}')

    # false positive y_gold=-1 y_pred=1

    # false negative y_gold=1 y_pred=-1


def cpe(xl, y, xu, alpha=1):
    c = np.sort(np.unique(y))
    n_c = len(c)

    x = []
    for i in range(n_c):
        x.append(xl[y == c[i], :])
    
    b = np.empty(n_c)
    for i in range(n_c):
        b[i] = 2*np.mean(alpha_dist(x[i], xu, alpha))

    A = np.empty((n_c, n_c))
    for i in range(n_c):
        for j in range(n_c):
            A[i, j] = -np.mean(alpha_dist(x[i], x[j], alpha))
    
    # T3 = np.mean(alpha_dist(xu, xu, alpha))

    As = A[:(n_c-1), :(n_c-1)]
    a = A[:(n_c-1), n_c-1]
    Ac = A[n_c-1, n_c-1]
    bs = b[:(n_c-1)]
    bc = b[n_c-1]

    Anew = ((As - a[:, None].T) - a) + Ac
    Anew = (Anew + Anew.T)/2
    bnew = 2*a - 2*Ac + bs - bc

    # x0 = -np.linalg.solve(2*Anew, bnew[:, None].T)
    x0 = -np.linalg.solve(2*Anew, bnew.T)
    x = np.minimum(np.maximum(0, x0), 1)
    theta = 1 - np.sum(x)

    return theta

@ex.command
def compute_cpe(corpus, dev_corpus=None, window_size=0, 
                pos_label=None, prior=0.5, sigma=.1, basis='gauss', 
                _run: Run = None, _log: logger = None):
    # labels = [-1, 1]
    _run.add_resource(corpus)
    X, y, tag = process_data(corpus, window_size, label_idx=1, 
                             pos_label=pos_label)
    if dev_corpus:
        X_dev, y_dev, tag_dev = process_data(dev_corpus, window_size, label_idx=1, 
                                             pos_label=pos_label, no_label=True)
        # print(list(zip(tag,y_dev)))
        X = np.concatenate((X, X_dev), axis=0)
        y = np.concatenate((y, y_dev), axis=0)
        tag = np.concatenate((tag, tag_dev), axis=0)

    tag = np.asanyarray(tag)
    clf = EntityDetectionPU(prior=prior, sigma=sigma, basis=basis)
    X = clf.featureizer.fit_transform(X, y)
    X = X.toarray()
    x_p, x_u = X[y == +1, :], X[y == 0, :]
    tag_p = tag[y == +1]
    theta = cpe(x_p, tag_p, x_u)
    _log.info(f'cpe theta: {theta}')
    _run.info['cpe theta'] = theta
    return theta

@ex.command
def train(train_corpus: str, dev_corpus: str, window_size: int = 1, 
          prior=0.5, sigma=.1, basis='gauss', pos_label=None,
          model_filename: str=None, _run: Run = None, _log: logger = None):
    """
    running pu learning entity detection training
    """
    _run.add_resource(train_corpus)
    _run.add_resource(dev_corpus)

    labels = [-1, 1]
    X_train, y_train, _ = process_data(train_corpus, window_size, label_idx=1, 
                                       pos_label=pos_label)
    X_dev, y_dev, tag_dev = process_data(dev_corpus, window_size, label_idx=1, 
                                         pos_label=pos_label, unlabeled=False)

    clf = EntityDetectionPU(prior=prior, sigma=sigma, basis=basis)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)

    sample_output(X_dev, y_dev, y_pred, tag_dev, _run, _log)
    
    evaluate_pu(y_dev, y_pred, labels, prior=prior, _run=_run, _log=_log)

    if model_filename is not None:
        _log.info(f'saving to: {model_filename}.pkl')
        joblib.dump(clf, f'{model_filename}.pkl')
        _run.add_artifact(f'{model_filename}.pkl')

@ex.command
def test(model_filename: str, test_corpus: str, window_size: int = 5, 
         prior=0.5, _run: Run = None, _log: logger = None):
    """
    running pu learning entity detection testing
    """
    _run.add_resource(test_corpus)
    _run.add_resource(f'{model_filename}.pkl')
    
    labels = [1, -1]
    X_test, y_test, _ = process_data(test_corpus, window_size, 1, unlabeled=False)

    _log.info(f'load from: {model_filename}.pkl')

    clf = joblib.load(f'{model_filename}.pkl')
    
    y_pred = clf.predict(X_test)

    evaluate_pu(y_test, y_pred, labels, prior=prior, _run=_run, _log=_log)

@ex.command
def hyperparams(train_files: List[str], 
                dev_files: List[str], 
                dirpath: str, 
                params_config_file: str,
                retry_limit: int = 10,
                _run: Run = None,
                _log: logger = logger):
    """
    running pu learning entity detection hyperparameter optimization
    """
    # absolute paths for all training sets
    train_corpora = [os.path.join(dirpath, t)  for t in train_files]
    # absolute paths for all dev sets
    dev_corpora = [os.path.join(dirpath, t) for t in dev_files]
    # absolute paths for all test sets
    params_def = yaml.load(open(params_config_file))
    configs = generate_params(params_def)

    for i, _ in enumerate(train_corpora):

        current_idx = 0
        current_ret = 0
        while current_idx < len(configs) and current_ret < retry_limit:
            try:
                config = configs[current_idx]
                config['train_corpus'] = train_corpora[i]
                config['dev_corpus'] = dev_corpora[i]

                _log.info(f'Run {current_idx + 1} of {len(configs)}')
                _log.info(f'Config: {config}')
                print(config)

                r = ex.run_command(command_name='train', config_updates=config)
                current_idx += 1
            except KeyboardInterrupt:
                _log.info('Experiment aborted')
                exit()
            except RuntimeError:
                if current_ret == retry_limit - 1:
                    logging.error('RETRY LIMIT EXCEEDED!, Experiment Failed')
                    break
                else:
                    fail = f'Run failed, will keep retrying {current_ret} of {retry_limit}'
                    _log.warning(fail)
                current_ret += 1
            except Exception:
                break

@ex.automain
def main():
    print('pu learning entity detection experiment main command.')
