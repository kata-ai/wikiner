
import os
import logging
from subprocess import run, PIPE, call, Popen
from typing import List
import tempfile

import scipy.stats
# TODO replace sklearn_crfsuite into partial_crfsuite
import sklearn_crfsuite

# TODO remove this joblib because partial-crf is called via command
from sklearn.externals import joblib
from sacred.observers import MongoObserver
from sacred import Experiment
from sacred.run import Run
from pymongo import MongoClient

from ingredients.crf_utils import get_tagged_sents_and_words
from ingredients.crf_utils import sent2features
from ingredients.crf_utils import sent2labels
from ingredients.crf_utils import sent2labels_colmap
from ingredients.crf_utils import sent2partial_labels
from ingredients.crf_utils import evaluate

SECRET = os.environ.get('SACRED_KEY', None)
MONGOL = f'mongodb://fariz:{SECRET}@ml-tools.kata.net:27017/sacredFariz'

ex = Experiment('run_crf_pa')

client = MongoClient(MONGOL)

ex.observers.append(MongoObserver.create(
    url=MONGOL, db_name='sacredFariz'))

db = client['sacredFariz']
runs = db['runs']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
logger.addHandler(handler)
ex.logger = logger

home = os.environ.get("HOME")

pre = ['B', 'I']
ent = ['Person', 'Place', 'Organisation']

@ex.config
def default_config():
    pacrf = f'{home}/partial-crf/frontend/crfsuite'
    train_files = ['train.conll']
    dev_files = ['dev.conll']
    test_files = ['test.conll']
    dirpath = '/home/fariz/workspace/dbpedia/dee'
    num_experiments = 30
    num_window_sizes = 4
    retry_limit = 10

    labels = [f'{x}-{y}' for x in pre for y in ent] + ['O']

@ex.named_config
def cross_val_config():
    train_files = [f'train-{i}.conll' for i in range(5)]
    dev_files = [f'test-{i}.conll' for i in range(5)]
    dirpath = '/home/fariz/workspace/dbpedia/dee'
    num_experiments = 30
    num_window_sizes = 5
    retry_limit = 10

@ex.named_config
def features_config():
    pass

@ex.capture
def print_corpus(corpus: List, labels: List, fileout: str, window_size: int = 0):
    for tagged_sent in corpus:
        feats = sent2features(tagged_sent, window_size)
        tags = sent2partial_labels(tagged_sent, labels=labels)
        for tag, feature in zip(tags, feats):
            feature = ['{}={}'.format(k, v) for k, v in feature.items()]
            print('{}\t{}'.format(tag, '\t'.join(feature)), file=fileout)
        print('',file=fileout)
    logging.info(f"print corpus to {fileout.name}")


def cmd_out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    return result.stdout


@ex.command
def train(train_corpus: str, dev_corpus: str, pacrf: str, 
          model_filename: str, labels: List, 
          c1: float = 0.0, c2: float = 1.0, 
          algorithm: str = 'lbfgs', max_iterations: int = None, 
          all_possible_transitions: bool = False, window_size: int = 0,
          _run: Run = None, _log: logger = None):
    """
    running crf experiment
    """
    _run.add_resource(train_corpus)
    _run.add_resource(dev_corpus)
    train_sents, _ = get_tagged_sents_and_words(train_corpus)
    dev_sents, _ = get_tagged_sents_and_words(dev_corpus)

    tmp_train = tempfile.NamedTemporaryFile(mode='w+')
    # temp_train_corpus = open(f'{model_filename}-{train_corpus}.feature', mode='w+')
    print_corpus(train_sents, labels, tmp_train, window_size=window_size)

    # X_dev = [sent2features(s, window_size) for s in dev_sents]
    y_dev = [sent2labels_colmap(s, col=1) for s in dev_sents]

    tmp_dev = tempfile.NamedTemporaryFile(mode='w+')
    # temp_test_corpus = open(f'{model_filename}-{test_corpus}.feature', mode='w+')
    print_corpus(dev_sents, labels, tmp_dev, window_size=window_size)

    # to call partial-crf via Popen command
    # command = f'{pacrf} learn -m {model_filename} -a {algorithm} {temp_train_corpus}'
    # call([pacrf, "--help"])

    crfsuire_proc = Popen([pacrf, "learn", "-m", model_filename, "-a", algorithm, \
                           "-p", f"c1={c1}", "-p", f"c2={c2}", tmp_train.name])
    out, err = crfsuire_proc.communicate()
    print(out)
    print(err)
    # os.system(f'{pacrf} learn -m {model_filename} -a {algorithm} {tmp_train.name}')

    tmp_train.close()

    tmp_pred = tempfile.NamedTemporaryFile(mode='w+')

    # cmd_out([pacrf, "tag", "-m", model_filename, tmp_dev.name, ">", tmp_pred.name])

    _run.add_artifact(model_filename)

    # TODO modified this to call partial-crf via Popen command
    # y_pred = crf.predict(X_dev)
    y_pred = get_tagged_sents_and_words(tmp_pred.name)
    print(y_pred)
    y_pred = [sent2labels_colmap(s, 0) for s in y_pred]


    # TODO modified this to read partial-crf via tempfile
    overall, by_type = evaluate(y_dev, y_pred)
    tmp_pred.close()
    tmp_dev.close()


    _run.info[f'overall_f1'] = overall.f1_score
    _run.log_scalar('overall_f1', overall.f1_score)
    _run.info[f'overall_precision'] = overall.precision
    _run.log_scalar('overall_precision', overall.precision)
    _run.info[f'overall_recall'] = overall.recall
    _run.log_scalar('overall_recall', overall.recall) 
    _log.info(f'Overall F1 score: {overall.f1_score}')
    for _, key in enumerate(sorted(by_type.keys())):
        for metric_key in by_type[key]._fields:
            metric_val = getattr(by_type[key], metric_key)
            _run.info[f'{key}-{metric_key}'] = metric_val
            _run.log_scalar(f'{key}-{metric_key}', metric_val)
            _log.info(f'{key}-{metric_key}: {metric_val}')


@ex.command
def test(model_filename: str, test_corpus: str, 
         pacrf: str, window_size: int = 5, 
         _run: Run = None, _log: logger = None):
    _run.add_resource(test_corpus)
    _run.add_resource(f'{model_filename}.pkl')
    test_sents, _ = get_tagged_sents_and_words(test_corpus)

    X_test = [sent2features(s, window_size) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    _log.info(f'load from: {model_filename}.pkl')

    # TODO modified this to call partial-crf via Popen command
    crf = sklearn_crfsuite.CRF(
            model_filename=model_filename
        )
    
    # TODO modified this to call partial-crf via Popen command
    y_pred = crf.predict(X_test)

    # TODO modified this to read partial-crf via tempfile
    overall, by_type = evaluate(y_test, y_pred)

    _run.info[f'overall_f1'] = overall.f1_score
    _run.log_scalar('overall_f1', overall.f1_score)
    _run.info[f'overall_precision'] = overall.precision
    _run.log_scalar('overall_precision', overall.precision)
    _run.info[f'overall_recall'] = overall.recall
    _run.log_scalar('overall_recall', overall.recall) 
    _log.info(f'Overall F1 score: {overall.f1_score}')
    for _, key in enumerate(sorted(by_type.keys())):
        for metric_key in by_type[key]._fields:
            metric_val = getattr(by_type[key], metric_key)
            _run.info[f'{key}-{metric_key}'] = metric_val
            _run.log_scalar(f'{key}-{metric_key}', metric_val)
            _log.info(f'{key}-{metric_key}: {metric_val}')


@ex.command
def hyperparams(train_files: List[str], 
                dev_files: List[str], 
                dirpath: str, 
                pacrf: str,
                num_experiments: int = 30, 
                num_window_sizes: int = 5, 
                retry_limit: int = 100, 
                _run: Run = None, _log: logger = None):
    """
    run hyperparameter optimization experiments
    """
    # absolute paths for all training sets
    train_corpora = [os.path.join(dirpath, t)  for t in train_files]
    # absolute paths for all dev sets
    dev_corpora = [os.path.join(dirpath, t) for t in dev_files]
    # absolute paths for all test sets
    for i, _ in enumerate(train_corpora):
        configs = []
        c1_space = scipy.stats.expon(scale=0.5)
        c2_space = scipy.stats.expon(scale=0.05)
        for _ in range(num_experiments):
            c1 = c1_space.rvs()
            c2 = c2_space.rvs()
            for window_size in range(0, num_window_sizes):
                configs.append({
                    'train_corpus': train_corpora[i],
                    'dev_corpus': dev_corpora[i],
                    'c1': c1, 
                    'c2': c2, 
                    'window_size': window_size
                })

        current_idx = 0
        current_ret = 0
        while current_idx < len(configs) and current_ret < retry_limit:
            try:
                logger.info(f'Run {current_idx + 1} of {len(configs)}')
                logger.info(f'Config: {configs[current_idx]}')
                r = ex.run_command(command_name='train', config_updates=configs[current_idx])
                current_idx += 1
            except KeyboardInterrupt:
                logger.info('Experiment aborted')
                exit()
            except RuntimeError:
                if current_ret == retry_limit - 1:
                    logging.error('RETRY LIMIT EXCEEDED!, Experiment Failed')
                    break
                else:
                    logger.warning(f'Run failed, will keep retrying {current_ret} of {retry_limit}')
                current_ret += 1
            except Exception:
                break

@ex.automain
def main():
    print('crf experiment main command.')
