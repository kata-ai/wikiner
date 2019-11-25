import logging
import os
from typing import List, Tuple, Optional

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models.archival import load_archive, Archive
from allennlp.models.model import Model
from allennlp.models.crf_tagger import CrfTagger


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("crf_tagger_char_pretrain")
class CrfTaggerCharPretrain(CrfTagger):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 constraint_type: str = None,
                 include_start_end_transitions: bool = True,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 initial_model_file: str = None) -> None:
        super(CrfTaggerCharPretrain, self).__init__(vocab=vocab, 
                                                text_field_embedder=text_field_embedder, 
                                                encoder=encoder, label_namespace=label_namespace,
                                                constraint_type=constraint_type, 
                                                include_start_end_transitions=include_start_end_transitions,
                                                dropout=dropout, 
                                                initializer=initializer, 
                                                regularizer=regularizer)

        if initial_model_file is not None:
            if os.path.isfile(initial_model_file):
                archive = load_archive(initial_model_file)
                self._initialize_weights_from_archive(archive)
            else:
                # A model file is passed, but it does not exist. This is expected to happen when
                # you're using a trained ERM model to decode. But it may also happen if the path to
                # the file is really just incorrect. So throwing a warning.
                logger.warning("initial model file for initializing weights is passed, but does not exist."
                               "This is fine if you're just predict or evaluate.")

    def _initialize_weights_from_archive(self, archive: Archive) -> None:
        logger.info("Initializing weights from pretrained model.")
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        # tokens_embedder_weight = "_text_field_embedder.token_embedder_tokens.weight"
        tokens_embedder_weight = "text_field_embedder.token_embedder_tokens.weight"
        if tokens_embedder_weight not in archived_parameters or \
           tokens_embedder_weight not in model_parameters:
            logger.info(archived_parameters.keys())
            raise RuntimeError("When initializing model weights from an CrfPretrain model, we need "
                               "the _text_field_embedder to be a TokenEmbedder using namespace called "
                               "tokens.")
        for name, weights in archived_parameters.items():
            if name in model_parameters:
                if name == "tag_projection_layer._module.weight" or name == "tag_projection_layer._module.bias"\
                    or name == "crf.transitions" or name == "crf._constraint_mask":
                    continue
                # if name == "_tag_projection_layer._output_projection_layer.weight":
                #     archived_projection_weights = weights.data
                #     new_weights = model_parameters[name].data.clone()
                #     new_weights[:, :-len(self._terminal_productions)] = archived_projection_weights
                if name == "text_field_embedder.token_embedder_tokens.weight":
                    # The shapes of embedding weights will most likely differ between the two models
                    # because the vocabularies will most likely be different. We will get a mapping
                    # of indices from this model's token indices to the archived model's and copy
                    # the tensor accordingly.
                    continue
                    # vocab_index_mapping = self._get_vocab_index_mapping(archive.model.vocab)
                    # archived_embedding_weights = weights.data
                    # new_weights = model_parameters[name].data.clone()
                    # for index, archived_index in vocab_index_mapping:
                    #     new_weights[index] = archived_embedding_weights[archived_index]
                    # logger.info("Copied embeddings of %d out of %d tokens",
                    #             len(vocab_index_mapping), new_weights.size()[0])
                elif name == "text_field_embedder.token_embedder_token_characters._embedding._module.weight":
                    vocab_index_mapping = self._get_vocab_index_mapping(archive.model.vocab, 'token_characters')
                    archived_embedding_weights = weights.data
                    new_weights = model_parameters[name].data.clone()
                    for index, archived_index in vocab_index_mapping:
                        new_weights[index] = archived_embedding_weights[archived_index]
                    logger.info("Copied embeddings of %d out of %d token_characters",
                                len(vocab_index_mapping), new_weights.size()[0])
                # elif name == "text_field_embedder.token_embedder_token_characters._encoder._module.conv_layer_0.weight":
                elif "text_field_embedder.token_embedder_token_characters._encoder._module.conv_layer" in name:
                    new_weights = weights.data
                else:
                    # new_weights = weights.data
                    continue
                logger.info("Copying parameter %s", name)
                model_parameters[name].data.copy_(new_weights)

    def _get_vocab_index_mapping(self, archived_vocab: Vocabulary, namespace='tokens') -> List[Tuple[int, int]]:
        vocab_index_mapping: List[Tuple[int, int]] = []
        for index in range(self.vocab.get_vocab_size(namespace=namespace)):
            token = self.vocab.get_token_from_index(index=index, namespace=namespace)
            archived_token_index = archived_vocab.get_token_index(token, namespace=namespace)
            # Checking if we got the UNK token index, because we don't want all new token
            # representations initialized to UNK token's representation. We do that by checking if
            # the two tokens are the same. They will not be if the token at the archived index is
            # UNK.
            if archived_vocab.get_token_from_index(archived_token_index, namespace=namespace) == token:
                vocab_index_mapping.append((index, archived_token_index))
        return vocab_index_mapping

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'CrfTaggerPretrain':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        label_namespace = params.pop("label_namespace", "labels")
        constraint_type = params.pop("constraint_type", None)
        dropout = params.pop("dropout", None)
        include_start_end_transitions = params.pop("include_start_end_transitions", True)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        initial_model_file = params.pop("initial_model_file", None)

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   label_namespace=label_namespace,
                   constraint_type=constraint_type,
                   dropout=dropout,
                   include_start_end_transitions=include_start_end_transitions,
                   initializer=initializer,
                   regularizer=regularizer,
                   initial_model_file=initial_model_file)
