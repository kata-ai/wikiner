import os
import logging
from typing import List

from pymongo import MongoClient
from sacred.observers import MongoObserver
from sacred import Experiment
from sacred.run import Run

SECRET = os.environ.get('SACRED_KEY', None)
MONGOL = f'mongodb://fariz:{SECRET}@ml-tools.kata.net:27017/sacredFariz'

ex = Experiment('run_stanford-crfsuite')

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

@ex.named_config
def cross_val_config():
    train_files = [f'train-{i}.conll' for i in range(5)]
    dev_files = [f'test-{i}.conll' for i in range(5)]
    dirpath = '/home/fariz/workspace/dbpedia/dee'

@ex.capture
def distill_crfsuite(models_name: List[str]):
    pass
