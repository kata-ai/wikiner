from typing import Optional, Dict
import logging
import torch
from torch.autograd.variable import Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import Linear

from overrides import overrides

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules import TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure

from ingredients.self_attn_mod import subsequent_mask
# from self_attn_mod import Embeddings
from ingredients.self_attn_mod import MultiHeadedAttention
from ingredients.self_attn_mod import PositionwiseFeedForward
from ingredients.self_attn_mod import PositionalEncoding
from ingredients.self_attn_mod import EncoderLayer, Encoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('transformer_tagger')
class Transformer(Model):
    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, n=6,
                 d_model=512, d_ff=2048, h=8, dropout=0.1, binary_feature_dim: int=10,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        super(Transformer, self).__init__(vocab=vocab, regularizer=regularizer)
        self.d_model = d_model + binary_feature_dim
        position = PositionalEncoding(d_model, dropout)
        self.src_embed = nn.Sequential(text_field_embedder, position)
        self.attn = MultiHeadedAttention(h, self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, d_ff, dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = Encoder(EncoderLayer(self.d_model, self.attn, self.ff, dropout), n)
        self.tags_proj = TimeDistributed(Linear(self.d_model, self.num_classes))
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels", ignore_classes=[])
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        initializer(self)

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    def forward(self, tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None):
        len_mask = get_text_field_mask(tokens)
        mask = self.make_std_mask(tags, len_mask)
        # logging.info(embedded_verb_indicator.size())
        feat = self.src_embed(tokens)
        # logging.info(mask.size())
        # logging.info(feat.size())
        batch_size, sequence_length, embedding_dim_with_binary_feature = feat.size()
        assert embedding_dim_with_binary_feature == self.d_model
        hidden = self.encoder(feat, mask)
        logits = self.tags_proj(hidden)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs)
        class_probabilities = class_probabilities.view([batch_size, sequence_length, self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, len_mask)
            self.span_metric(class_probabilities, tags, len_mask)
            output_dict["loss"] = loss

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = len_mask
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.span_metric.get_metric(reset=reset)
        if self.training:
            # This can be a lot of metrics, as there are 3 per class.
            # During training, we only really care about the overall
            # metrics, so we filter for them here.
            # TODO(Mark): This is fragile and should be replaced with some verbosity level in Trainer.
            return {x: y for x, y in metric_dict.items() if "overall" in x}

        return metric_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].data.cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        binary_feature_dim = params.pop("binary_feature_dim")
        n = params.pop("n", 6)
        d_model = params.pop("d_model", 512)
        d_ff = params.pop("d_ff", 2048)
        h = params.pop("h", 8)
        dropout = params.pop("dropout", 0.1)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   binary_feature_dim=binary_feature_dim,
                   n=n, d_model=d_model, d_ff=d_ff, h=h,
                   dropout=dropout,
                   initializer=initializer,
                   regularizer=regularizer)
