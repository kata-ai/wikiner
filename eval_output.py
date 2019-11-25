import os
import sys
import logging
from itertools import zip_longest

from pymongo import MongoClient
from sacred.observers import MongoObserver
from sacred import Experiment
from sacred.run import Run

from ingredients.crf_utils import get_tagged_sents_and_words
from ingredients.crf_utils import sent2labels_colmap
from ingredients.crf_utils import evaluate

SECRET = os.environ.get('SACRED_KEY', None)
MONGOL = f'mongodb://fariz:{SECRET}@ml-tools.kata.net:27017/sacredFariz'

ex = Experiment('eval_output')

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

@ex.automain
def test(test_corpus: str, model_output: str,
         col_ref: int = 0, col_hyp: int = 0,
         _run: Run = None, _log: logger = None):
    test_sents, _ = get_tagged_sents_and_words(test_corpus)
    print(f'num sentences: {len(test_sents)}')
    y_test = [sent2labels_colmap(s, col=int(col_ref)) for s in test_sents]


    yout_sents, _ = get_tagged_sents_and_words(model_output)
    print(f'num sentences: {len(yout_sents)}')
    y_pred = [sent2labels_colmap(s, col=int(col_hyp)) for s in yout_sents]

    if len(y_test) != len(y_pred):
        for i, j in zip_longest(y_test, y_pred):
            print(i, j)

    overall, by_type = evaluate(y_test, y_pred)
    print(overall)
    print(by_type)

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
