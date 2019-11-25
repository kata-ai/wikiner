from typing import List
import os
import logging

import yaml
from sacred.observers import MongoObserver
from sacred import Experiment
from sacred.run import Run
from pymongo import MongoClient

from seqlab.models import SequenceLabeler
from ingredients.seqlab_mod import NewSequenceLabeler
from ingredients.seqlab_mod import TrainerMod
from ingredients.seqlab_mod import EvaluatorMod
from ingredients.gen_configs import generate_params

SECRET = os.environ.get('SACRED_KEY', None)
MONGOL = f'mongodb://fariz:{SECRET}@ml-tools.kata.net:27017/sacredFariz'

ex = Experiment('run_seqlab')

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


@ex.command
def train(train_corpus: str, dev_corpus: str, 
          char_int: int, save_path: str, 
          test_corpus: str = None, dropout: float = 0.5, 
          num_epochs: int = 10, lm_loss_scale=0.1, device: int = 0, 
          save=False, _run: Run = None):
    _run.add_resource(train_corpus)
    _run.add_resource(dev_corpus)
    trainer = TrainerMod(_run, train_corpus, save_path, dev_corpus, 
                         num_epochs=num_epochs, dropout=dropout, 
                         char_integration_method=char_int, 
                         lm_loss_scale=lm_loss_scale, save=save,
                         device=device)
    trainer.run()
    if test_corpus:
        _run.add_resource(test_corpus)
        ex.run_command('test', config_updates={
            'save_path': save_path,
            'test_corpus': test_corpus,
            'device': device
        })

@ex.command
def train_w_pretrained(train_corpus: str, dev_corpus: str, 
                       char_int: int, pretrained_embeddings: str, save_path: str, 
                       test_corpus: str = None, word_embedding_size: int = 300, 
                       update_pretrained_embedding: bool = True,
                       dropout: float = 0.5, num_epochs: int = 10, lm_loss_scale=0.1, 
                       device: int = 0, save=False, _run: Run = None):
    _run.add_resource(train_corpus)
    _run.add_resource(dev_corpus)
    trainer = TrainerMod(_run, train_corpus, save_path, dev_corpus,
                         word_embedding_size=word_embedding_size, 
                         num_epochs=num_epochs, dropout=dropout, 
                         char_integration_method=char_int, 
                         lm_loss_scale=lm_loss_scale, save=save,
                         device=device, pretrained_embeddings=pretrained_embeddings,
                         update_pretrained_embedding=update_pretrained_embedding,
                         model_class=NewSequenceLabeler)
    trainer.run()
    if test_corpus:
        _run.add_resource(test_corpus)
        ex.run_command('test_w_pretrained', config_updates={
            'save_path': save_path,
            'test_corpus': test_corpus,
            'device': device
        })

@ex.command
def test_w_pretrained(save_path: str, test_corpus: str, device: int = -1, _run: Run = None):
    evaluator = EvaluatorMod(_run, artifacts_path=save_path + '/artifacts.tar.gz',
                             corpus_path=test_corpus,
                             device=device, model=NewSequenceLabeler)
    evaluator.run()

@ex.command
def test(save_path: str, test_corpus: str, device: int = -1, _run: Run = None):
    evaluator = EvaluatorMod(_run, artifacts_path=save_path + '/artifacts.tar.gz',
                             corpus_path=test_corpus,
                             device=device, model=SequenceLabeler)
    evaluator.run()
    

@ex.command
def hyperparams(train_files: List[str], 
                dev_files: List[str], 
                dirpath: str, 
                params_config_file: str,
                retry_limit: int = 10,
                _run: Run = None):
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

                logger.info(f'Run {current_idx + 1} of {len(configs)}')
                logger.info(f'Config: {config}')
                print(config)

                r = ex.run_command(command_name='train', config_updates=config)
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
    print('seqlab experiment main command.')
