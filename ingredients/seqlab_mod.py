from typing import List, Optional, Tuple, Union
import os
import tempfile
import tarfile
import json

import dill
import torch
from torch.autograd import Variable
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchcrf import CRF
from torchtext.vocab import Vectors
from sacred.run import Run

from seqlab.models import SequenceLabeler
from seqlab.trainer import Trainer
from seqlab.evaluator import Evaluator

from helfer.evaluation import evaluate_conll

class NewSequenceLabeler(SequenceLabeler):
    def __init__(self,
                 num_words: int,
                 num_tags: int,
                 num_chars: int = 0,
                 word_embedding_size: int = 300,
                 dropout: float = 0,
                 var_dropout: float = 0,
                 embed_dropout: float = 0,
                 use_pretrained_embeddings=False,
                 pretrained_emb_dir='',
                 update_pretrained_embedding=True,
                 encoder_hidden_size: int = 200,
                 lm_loss_scale: float = 0.1,
                 lm_layer_size: int = 50,
                 lm_max_vocab_size: int = 7500,
                 pre_output_layer_size: int = 50,
                 char_integration_method: str = 'none',
                 char_embedding_size: int = 50,
                 char_encoder_hidden_size: int = 200,
                 unk_id: int = 0,
                 ) -> None:
        if num_words <= 0:
            raise ValueError(f'invalid number of words: {num_words}')
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        if word_embedding_size <= 0:
            raise ValueError(
                f'invalid word embedding size: {word_embedding_size}')
        if dropout < 0. or dropout >= 1.:
            raise ValueError(f'invalid dropout rate: {dropout:.2f}')
        if encoder_hidden_size <= 0:
            raise ValueError(
                f'invalid encoder hidden size: {encoder_hidden_size}')
        if lm_layer_size <= 0:
            raise ValueError(
                f'invalid language modeling layer size: {lm_layer_size}')
        if lm_max_vocab_size <= 0:
            raise ValueError(
                f'invalid language modeling max vocabulary size: {lm_max_vocab_size}')
        if char_integration_method not in ('none', 'concatenation', 'attention'):
            raise ValueError(
                f'invalid character integration method: {char_integration_method}')
        if char_embedding_size <= 0:
            raise ValueError(
                f'invalid character embedding size: {char_embedding_size}')
        if char_encoder_hidden_size <= 0:
            raise ValueError(
                f'invalid character encoder hidden size: {char_encoder_hidden_size}')

        # These two conditionals ensure that
        # char_integration_method == 'none' <-> not num_chars
        if char_integration_method != 'none' and not num_chars:
            raise ValueError(
                "'num_chars' must be specified when integration method is not 'none'")
        if char_integration_method == 'none' and num_chars:
            raise ValueError(
                f"invalid value for 'num_chars' when integration method is 'none': {num_chars}")
        assert (char_integration_method != 'none' or not num_chars) and \
            (char_integration_method == 'none' or num_chars)

        self.use_pretrained_embeddings = use_pretrained_embeddings

        super(NewSequenceLabeler, self).__init__(num_words=num_words, 
                                                 num_tags=num_tags, 
                                                 num_chars=num_chars,
                                                 word_embedding_size=word_embedding_size,
                                                 dropout=dropout, 
                                                 encoder_hidden_size=encoder_hidden_size,
                                                 lm_loss_scale=lm_loss_scale, lm_layer_size=lm_layer_size,
                                                 lm_max_vocab_size=lm_max_vocab_size, 
                                                 pre_output_layer_size=pre_output_layer_size, 
                                                 char_integration_method=char_integration_method,
                                                 char_embedding_size=char_embedding_size, 
                                                 char_encoder_hidden_size=char_encoder_hidden_size, 
                                                 unk_id=unk_id)

        # # Attributes
        # self.num_words = num_words
        # self.num_tags = num_tags
        # self.num_chars = num_chars
        # self.word_embedding_size = word_embedding_size
        # self.var_dropout = var_dropout
        # self.embed_dropout = embed_dropout
        # self.dropout = dropout
        # self.encoder_hidden_size = encoder_hidden_size
        # self.lm_loss_scale = lm_loss_scale
        # self.lm_layer_size = lm_layer_size
        # self.lm_max_vocab_size = lm_max_vocab_size
        # self.pre_output_layer_size = pre_output_layer_size
        # self.char_integration_method = char_integration_method
        # self.char_embedding_size = char_embedding_size
        # self.char_encoder_hidden_size = char_encoder_hidden_size
        # self.unk_id = unk_id

        # Embeddings
        self.word_embedding = nn.Embedding(num_words, word_embedding_size)
        if use_pretrained_embeddings:
            pretrained_embed_weights = torch.load(
                os.path.join(pretrained_emb_dir, 'pretrained_embed_weights.pt'))
            for i, _ in enumerate(pretrained_embed_weights):
                if len(pretrained_embed_weights[i, :].nonzero()) == 0:
                    torch.nn.init.normal(pretrained_embed_weights[i], std=0.1)
            self.word_embedding.weight.data.copy_(pretrained_embed_weights)
            if not(update_pretrained_embedding):
                self.word_embedding.requires_grad = False

        if self.uses_char_embeddings:
            self.char_embedding = nn.Embedding(num_chars, char_embedding_size)

        # # Char encoder LSTM
        # if self.uses_char_embeddings:
        #     self.char_encoder = nn.LSTM(
        #         char_embedding_size,
        #         char_encoder_hidden_size,
        #         num_layers=1,
        #         batch_first=True,
        #         dropout=0.,
        #         bidirectional=True,
        #     )
        #     self.char_projection = nn.Sequential(
        #         nn.Linear(2 * char_encoder_hidden_size, word_embedding_size),
        #         nn.Tanh(),
        #     )

        # # Attention
        # if char_integration_method == 'attention':
        #     self.attention = nn.Sequential(
        #         nn.Linear(2 * word_embedding_size, word_embedding_size),
        #         nn.Tanh(),
        #         nn.Linear(word_embedding_size, word_embedding_size),
        #         nn.Sigmoid(),
        #     )

        # # Seq2seq encoder
        # if self.dropout:
        #     self.dropout_layer = nn.Dropout(dropout)
        # encoder_input_size = word_embedding_size
        # if char_integration_method == 'concatenation':
        #     encoder_input_size *= 2

        # self.encoder = nn.LSTM(
        #     encoder_input_size,
        #     encoder_hidden_size,
        #     num_layers=1,
        #     batch_first=True,
        #     dropout=0.,
        #     bidirectional=True,
        # )

        # # Language modeling
        # if self.uses_lm_loss:
        #     lm_output_size = min(num_words, lm_max_vocab_size) + 1
        #     self.lm_ff_fwd = nn.Sequential(
        #         nn.Linear(encoder_hidden_size, lm_layer_size),
        #         nn.Tanh(),
        #         nn.Linear(lm_layer_size, lm_output_size),
        #     )
        #     self.lm_ff_bwd = nn.Sequential(
        #         nn.Linear(encoder_hidden_size, lm_layer_size),
        #         nn.Tanh(),
        #         nn.Linear(lm_layer_size, lm_output_size),
        #     )

        # # Output layer
        # if self.uses_pre_output_layer:
        #     self.pre_output_layer = nn.Sequential(
        #         nn.Linear(2 * encoder_hidden_size, pre_output_layer_size),
        #         nn.Tanh(),
        #     )
        #     self.output_layer = nn.Linear(pre_output_layer_size, num_tags)
        # else:
        #     self.output_layer = nn.Linear(2 * encoder_hidden_size, num_tags)
        # self.crf = CRF(num_tags)

        # self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all model parameters.

        As implemented by Rei et al., all parameters are initialized randomly
        from normal distribution with mean 0 and stddev 0.1.
        """
        for name, param in self.named_parameters():
            if not (name == 'word_embedding.weight' and self.use_pretrained_embeddings):
                nn.init.normal(param, std=0.1)

    def _compute_emissions_and_loss(self,
                                    words: Variable,
                                    chars: Variable,
                                    mask: Variable,
                                    ) -> Tuple[Variable, Variable]:

            # words: (batch_size, seq_length)
            # chars: (batch_size, seq_length, char_seq_length)
            # mask: (batch_size, seq_length, char_seq_length)

        assert words.dim() == 2
        assert chars.dim() == 3
        assert chars.size() == mask.size()

        loss = 0.

        # (batch_size, seq_length, word_emb_size)

        embedded_words = self.word_embedding(words)

        if self.uses_char_embeddings:
            # (batch_size, seq_length, word_emb_size)
            encoded_chars = self._compose_char_embeddings(chars, mask)
            if self.char_integration_method == 'concatenation':
                # (batch_size, seq_length, word_emb_size * 2)
                inputs = torch.cat([embedded_words, encoded_chars], dim=-1)
            else:  # must be attention
                # Add cosine similarity loss for non-unk words
                loss += self._compute_similarity_loss(
                    words, embedded_words, encoded_chars)
                # Compute attention weights
                # (batch_size, seq_length, word_emb_size * 2)
                concatenated = torch.cat(
                    [embedded_words, encoded_chars], dim=-1)
                # (batch_size, seq_length, word_emb_size)
                z = self.attention(concatenated)
                # (batch_size, seq_length, word_emb_size)
                inputs = embedded_words*z + encoded_chars*(1.-z)
        else:
            # (batch_size, seq_length, word_emb_size)
            inputs = embedded_words

        # (batch_size, seq_length, hidden_size * 2)
        if self.dropout:
            encoded, _ = self.encoder(self.dropout_layer(inputs))
        else:
            encoded, _ = self.encoder(inputs)

        if self.uses_lm_loss:
            # Add language modeling loss; the loss is summed over batch
            loss += self.lm_loss_scale * self._compute_lm_loss(encoded, words)

        if self.uses_pre_output_layer:
            # (batch_size, seq_length, pre_output_layer_size)
            encoded = self.pre_output_layer(encoded)
        # (batch_size, seq_length, num_tags)
        outputs = self.output_layer(encoded)
        # Remove start and end token
        # (batch_size, seq_length - 2, num_tags)
        outputs = outputs[:, 1:-1, :]

        return outputs, loss

    def forward(self,
                words: Variable,
                chars: Variable,
                tags: Optional[Variable] = None,
                mask: Optional[Variable] = None,
                ) -> Variable:
        """Compute the loss of the given batch of sentences and tags.

        Arguments
        ---------
        words : :class:`~torch.autograd.Variable`
            Word indices tensor of type ``LongTensor`` and size ``(batch_size, seq_length)``.
            This tensor should include the start and end token indices.
        chars : :class:`~torch.autograd.Variable`
            Character indices tensor of type ``LongTensor`` and size ``(batch_size, seq_length,
            char_seq_length)``. This tensor should include the character indices of the start
            and end token.
        tags : :class:`~torch.autograd.Variable`
            Tag indices tensor of type ``LongTensor`` and size ``(batch_size, seq_length - 2)``.
            This tensor should *not* include the tag for the start and end token.
        mask : :class:`~torch.autograd.Variable`, optional
            Mask tensor of type ``ByteTensor`` and size ``(batch_size, seq_length,
            char_seq_length)`` indicating the valid entries in ``chars``.

        Returns
        -------
        :class:`~torch.autograd.Variable`
            A variable of type ``FloatTensor`` and size ``(1,)`` containing the loss, summed
            over batch.
        """

        self._check_inputs_dimensions_and_sizes(
            words, chars, tags=tags, mask=mask)
        if mask is None:
            mask = Variable(self._new(chars.size()).fill_(1)).byte()

        outputs, loss = self._compute_emissions_and_loss(words, chars, mask)

        # Transpose batch_size and seq_length for CRF
        # NOTE transposing tensors make them not contiguous, but CRF needs them so
        # (seq_length - 2, batch_size, num_tags)
        outputs = outputs.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        if tags is not None:
            # Return loss minus CRF log likelihood
            loss = loss - self.crf(outputs, tags)
        return loss


class TrainerMod(Trainer):
    """Trainer for sequence labeler model.

    This class serves as a trainer for the sequence labeler model. At the heart of this class
    is its entrypoint, the ``run`` method, which implements a `template method pattern`_. The
    method executes several steps during training where individual step is implemented as
    a method. Thus, customizing the training algorithm can be done by simply inheriting from
    this class and overriding the desired methods.

    Arguments
    ---------
    train_corpus : str
        Path to the training corpus in two-column CoNLL format.
    save_to : str
        Path to a directory to save training artifacts.
    dev_corpus : str, optional
        Path to the development corpus in two-column CoNLL format.
    encoding : str
        File encoding to use. (default: utf-8)
    min_freq : int
        Minimum frequency of a word to be included in the vocabulary. (default: 2)
    word_embedding_size : int
        Size of word embeddings. (default: 300)
    dropout : float
        The dropout rate. (default: 0.5)
    encoder_hidden_size : int
        Size of the LSTM encoder hidden layer. (default: 200)
    lm_loss_scale : float
        Scaling coefficient for the language modeling loss. If set to a nonpositive scalar,
        language modeling loss is not computed. (default: 0.1)
    lm_layer_size : int
        Hidden layer size for language modeling. (default: 50)
    lm_max_vocab_size : int
        Maximum vocabulary size for language modeling. (default: 7500)
    pre_output_layer_size : int
        Size of pre-output layer. If set to a nonpositive number, no pre-output layer is
        used. (default: 50)
    char_integration_method : str
        How to integrate character embeddings. Possible values are 'none', 'concatenation',
        and 'attention'. If ``num_chars`` is 0 then this must be 'none', and vice versa.
        (default: 'none')
    char_embedding_size : int
        Size of character embeddings. (default: 50)
    char_encoder_hidden_size : int
        Size of the LSTM character encoder hidden layer. (default: 200)
    learning_rate : float
        The learning rate. (default: 0.001)
    optimizer : str
        Name of optimizer to use. Can be 'adam' or 'sgd'. (default: adam)
    num_epochs : int
        Number of epochs to train. (default: 10)
    batch_size : int
        Number of samples in one batch. (default: 16)
    device : int
        GPU device to use. Set to -1 for CPU. (default: -1)
    log_interval : int
        Print log every this number of updates. (default: 10)
    seed : int
        Random seed. (default: 201)
    logger : `~logging.Logger`, optional
        Logger object to use for logging.

    .. _template method pattern: https://en.wikipedia.org/wiki/Template_method_pattern
    """
    def __init__(self, _run : Run,
                 train_corpus: str,
                 save_to: str,
                 dev_corpus: str,
                 num_epochs: int,
                 char_integration_method: str,
                 word_embedding_size: int = 300,
                 lm_loss_scale: float = 0.1,
                 dropout: float = 0.,
                 pretrained_embeddings: str = None,
                 update_pretrained_embedding: bool = True,
                 learning_rate: float = 0.001,
                 decay: bool = False,
                 device: int = -1,
                 decay_patience: int = 10,
                 stop_patience: int = 3,
                 early_stopping: bool = True,
                 save: bool = False,
                 model_class = SequenceLabeler) -> None:
        torch.backends.cudnn.enabled = False

        self.sacred_run = _run  # instance of the experiment's sacred run object
        self.stop_patience = stop_patience
        self.decay_patience =decay_patience
        self.decay = decay
        self.best_loss = 1e15
        self.early_stopping = early_stopping
        self.dev_corpus = dev_corpus
        self.save = save
        self.pretrained_embeddings = pretrained_embeddings
        self.update_pretrained_embedding = update_pretrained_embedding
        self.model_class = model_class
        super(TrainerMod, self).__init__(train_corpus,
                                         save_to,
                                         self.dev_corpus,
                                         word_embedding_size=word_embedding_size,
                                         num_epochs=num_epochs,
                                         dropout=dropout,
                                         lm_loss_scale=lm_loss_scale,
                                         char_integration_method=char_integration_method,
                                         learning_rate=learning_rate,
                                         device=device)

    def on_start(self, state: dict) -> None:
        if state['train']:
            self.logger.info('Start training')
            self.train_timer.reset()
            # decaying learning rate
            if(self.decay):
                self.scheduler = ReduceLROnPlateau(
                                    self.optimizer, 
                                    patience=self.decay_patience, 
                                    factor=0.5)
        else:
            self.reset_meters()
            self.model.eval()

    def on_end_epoch(self, state: dict) -> None:
        elapsed_time = self.epoch_timer.value()
        assert len(self.references) == len(self.train_dataset)
        overall, by_type = self.get_conll_evaluation()
        self.logger.info(
            'Epoch %d done (%.4fs): %.4f samples/s | loss %.4f | F1 %.4f',
            state['epoch'], elapsed_time, self.speed_meter.value()[0],
            self.loss_meter.value()[0], overall.f1_score)

        self.sacred_run.log_scalar(
            'train_loss', self.loss_meter.value()[0], state['epoch'])
        self.sacred_run.log_scalar(
            'train_f1', overall.f1_score, state['epoch'])

        if(self.decay):
            self.scheduler.step(self.loss_meter.value()[0])  # lr decay scheduler

        self.logger.info(self.format_per_tag_f1_score(by_type))
        self.save_model()

        if self.dev_iterator is not None:
            self.logger.info('Evaluating on dev set')
            self.engine.test(self.network, self.dev_iterator)
            assert len(self.references) == len(self.dev_iterator.dataset)
            overall, by_type = self.get_conll_evaluation()
            self.logger.info(
                'Result on dev set: %.4f samples/s | loss %.4f | F1 %.4f',
                self.speed_meter.value()[0], self.loss_meter.value()[0],
                overall.f1_score)
            self.sacred_run.log_scalar(
                'dev_loss', self.loss_meter.value()[0], state['epoch'])
            self.sacred_run.log_scalar(
                'dev_f1', overall.f1_score, state['epoch'])
            self.current_loss = self.loss_meter.value()[0]
            if(self.early_stopping):
                if self.current_loss < self.best_loss:
                    self.best_loss = self.current_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.stop_patience:
                        self.logger.info(
                            f"Early stop triggered after {state['epoch']} epoch(s)")
                        # terminating condition for train-loop in tnt
                        state['maxepoch'] = state['epoch']
            for i, key in enumerate(sorted(by_type.keys())):
                score = by_type[key].f1_score
                score = f'{score:.4f}'
                self.sacred_run.log_scalar(key + "_f1", score, state['epoch'])

            self.logger.info(self.format_per_tag_f1_score(by_type))

    def save_artifacts(self) -> None:
        if self.save:
            super().save_artifacts()
        else:
            pass

    def build_model(self) -> None:
        model_args = (self.num_words, self.num_tags)
        print(model_args)
        num_chars = 0 if self.char_integration_method == 'none' else self.num_chars
        model_kwargs = dict(
            num_chars=num_chars,
            word_embedding_size=self.word_embedding_size,
            dropout=self.dropout,
            encoder_hidden_size=self.encoder_hidden_size,
            lm_loss_scale=self.lm_loss_scale,
            lm_layer_size=self.lm_layer_size,
            lm_max_vocab_size=self.lm_max_vocab_size,
            pre_output_layer_size=self.pre_output_layer_size,
            char_integration_method=self.char_integration_method,
            char_embedding_size=self.char_embedding_size,
            char_encoder_hidden_size=self.char_encoder_hidden_size,
            unk_id=self.WORDS.vocab.stoi[self.WORDS.unk_token]
        )

        if self.model_class == NewSequenceLabeler:
            use_pretrained_embeddings = True if self.pretrained_embeddings else False
            model_kwargs['use_pretrained_embeddings'] = use_pretrained_embeddings
            model_kwargs['update_pretrained_embedding'] = self.update_pretrained_embedding
            model_kwargs['pretrained_emb_dir'] = self.save_to

        self.model = self.model_class(*model_args, **model_kwargs)
        if self.device >= 0:
            self.model.cuda(self.device)
        self.logger.info('Saving model metadata to %s',
                         self.model_metadata_path)
        with open(self.model_metadata_path, 'w') as f:
            json.dump({'args': model_args, 'kwargs': model_kwargs},
                      f, indent=2, sort_keys=True)
        self.save_model()

    # modified for pretrained embedding
    def build_vocabularies(self) -> None:
        self.logger.info('Building vocabularies')
        if self.pretrained_embeddings is not None:
            vectors = Vectors(self.pretrained_embeddings)
            self.WORDS.build_vocab(
                self.train_dataset, min_freq=self.min_freq, vectors=vectors)
            vecfile = os.path.join(self.save_to, 'pretrained_embed_weights.pt')
            torch.save(self.WORDS.vocab.vectors, vecfile)
        else:
            self.WORDS.build_vocab(self.train_dataset, min_freq=self.min_freq)
        self.CHARS.build_vocab(self.train_dataset)
        self.TAGS.build_vocab(self.train_dataset)

        self.num_words = len(self.WORDS.vocab)
        self.num_chars = len(self.CHARS.vocab)
        self.num_tags = len(self.TAGS.vocab)
        self.logger.info(
            'Found %d words, %d chars, and %d tags',
            self.num_words, self.num_chars, self.num_tags)

        self.logger.info('Saving fields to %s', self.fields_path)
        torch.save(self.fields, self.fields_path, pickle_module=dill)

    def run(self) -> None:
        self.set_random_seed()
        self.prepare_for_serialization()
        self.init_fields()
        self.process_corpora()
        self.build_vocabularies()
        self.build_model()
        self.build_optimizer()

        self.engine.hooks['on_start'] = self.on_start
        self.engine.hooks['on_start_epoch'] = self.on_start_epoch
        self.engine.hooks['on_sample'] = self.on_sample
        self.engine.hooks['on_forward'] = self.on_forward
        self.engine.hooks['on_end_epoch'] = self.on_end_epoch
        self.engine.hooks['on_end'] = self.on_end

        try:
            self.engine.train(
                self.network, self.train_iterator, self.num_epochs, self.optimizer)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted, aborting')
            self.save_artifacts()

class EvaluatorMod(Evaluator):
    def __init__(self, _run, model, artifacts_path: str, corpus_path: str, device=-1
                 ) -> None:
        self.sacred_run = _run  # instance of the experiment's sacred run object
        self.artifacts_path = artifacts_path
        self.corpus_path = corpus_path
        self.model = model
        super().__init__(artifacts_path, corpus_path, device=device)

    def report_evaluation(self) -> None:
        self.logger.info('Evaluating hypotheses against references')
        evaluable = [zip(*sent_pair)
                     for sent_pair in zip(self.references, self.hypotheses)]
        overall, by_type = evaluate_conll(evaluable)
        self.sacred_run.info['overall_precision'] = overall.precision
        self.sacred_run.info['overall_recall'] = overall.recall
        self.sacred_run.info['overall_f1'] = overall.f1_score
        # for key in sorted(by_type.keys()):
        #     score = by_type[key].f1_score
        #     self.sacred_run.info[key] = score
        for _, key in enumerate(sorted(by_type.keys())):
            for metric_key in by_type[key]._fields:
                metric_val = getattr(by_type[key], metric_key)
                self.sacred_run.info[f'{key}-{metric_key}'] = metric_val
                self.sacred_run.log_scalar(f'{key}-{metric_key}', metric_val)

    def load_artifacts(self, model) -> None:
        if self.artifacts_loaded:
            return

        self.logger.info('Loading artifacts from %s', self.artifacts_path)
        artifact_names = [
            TrainerMod.FIELDS_FILENAME,
            TrainerMod.MODEL_METADATA_FILENAME,
            TrainerMod.MODEL_PARAMS_FILENAME,
        ]
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.logger.info('Extracting artifacts to %s', tmpdirname)
            with tarfile.open(self.artifacts_path, 'r:gz') as f:
                members = [member for member in f.getmembers()
                           if member.name in artifact_names]
                f.extractall(tmpdirname, members=members)

            self.logger.info('Loading fields')
            self.fields = torch.load(
                os.path.join(tmpdirname, TrainerMod.FIELDS_FILENAME), pickle_module=dill)
            fields_dict = dict(self.fields)
            self.WORDS = fields_dict[TrainerMod.WORDS_FIELD_NAME]
            self.CHARS = fields_dict[TrainerMod.CHARS_FIELD_NAME]
            self.TAGS = fields_dict[TrainerMod.TAGS_FIELD_NAME]

            self.logger.info('Loading model metadata')
            with open(os.path.join(tmpdirname, TrainerMod.MODEL_METADATA_FILENAME)) as fm:
                self.model_metadata = json.load(fm)

            self.logger.info('Building model and restoring model parameters')
            self.model = model(
                *self.model_metadata['args'], **self.model_metadata['kwargs'])
            # Load to CPU, 
            # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/4  
            # noqa
            self.model.load_state_dict(
                torch.load(os.path.join(tmpdirname, TrainerMod.MODEL_PARAMS_FILENAME),
                           map_location=lambda storage, loc: storage))
            if self.device >= 0:
                self.model.cuda(self.device)
            self.artifacts_loaded = True

    def run(self) -> None:
        self.load_artifacts(self.model)
        self.load_corpus()
        self.compute_references_and_hypotheses()
        self.report_evaluation()
