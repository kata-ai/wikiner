# pylint: disable=wrong-import-position,W0611
import logging
import os
# import sys


# sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

# from allennlp.models import CrfTagger

from allennlp.commands import main

from ingredients.custom_conll import CustomConll
from ingredients.wpne_readers import WordTagTupleReader
from ingredients.winer_readers import TargzReaders
from ingredients.multi_reader import MultiCorpusReader
from ingredients.learning_rate import SlantedTriangular
from ingredients.self_attn_mod import *
from ingredients.self_attn import Transformer
from ingredients.crf_multilingual import CrfTaggerCrossLingual
from ingredients.crf_pretrain import CrfTaggerPretrain
from ingredients.crf_char_trans import CrfTaggerCharPretrain
from ingredients.map_tagger import MapTagger

from ingredients.conll_predictor import TokenizedTaggerPredictor
from ingredients.conll_predict import ConllPredict, DEFAULT_PREDICTORS

DEFAULT_PREDICTORS['crf_tagger_pretrain'] = 'sentence-tagger'

subcommand_map = {
    'conll-output': ConllPredict(DEFAULT_PREDICTORS)
}

if __name__ == "__main__":
    # logging.warning("run is deprecated, please use allenrun.py")
    main(subcommand_overrides=subcommand_map)
