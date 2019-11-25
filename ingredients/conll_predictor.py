from typing import Tuple
from overrides import overrides

from allennlp.data.tokenizers import Token

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('tokenized-tagger')
class TokenizedTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        # sentence = json_dict["sentence"]
        # TODO change this into no tokens predictor
        # tokens = self._tokenizer.split_words(sentence)
        tokens = json_dict["sentence"]
        if not(isinstance(tokens, list)):
            raise ValueError(f'input {tokens} must be tokenized into list of tokens')
        tokens = [Token(token) for token in tokens]
        instance = self._dataset_reader.text_to_instance(tokens)

        return_dict: JsonDict = {"words":[token.text for token in tokens]}

        return instance, return_dict
