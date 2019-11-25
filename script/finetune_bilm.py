
import os
import time
import json
import re
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer

from bilm.training import load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset

from bilm.data import Vocabulary
from bilm.data import UnicodeCharsVocabulary
from bilm.data import InvalidNumberOfCharacters

from bilm.training import LanguageModel
from bilm.training import _get_feed_dict_from_X
from bilm.training import print_variable_summary
from bilm.training import average_gradients
from bilm.training import clip_grads
from bilm.training import summary_gradient_updates


DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)

def finetune(options, data, n_gpus, tf_save_dir, tf_log_dir,
             restart_ckpt_file=None):

    # not restarting so save the options
    if restart_ckpt_file is None:
        with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
            fout.write(json.dumps(options))

    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # set up the optimizer
        lr = options.get('learning_rate', 0.2)
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)

        # calculate the gradients on each GPU
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        norm_summaries = []
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.variable_scope('lm', reuse=k > 0):
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = LanguageModel(options, True)
                    loss = model.total_loss
                    models.append(model)
                    # get gradients
                    grads = opt.compute_gradients(
                        loss * options['unroll_steps'],
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grads.append(grads)
                    # keep track of loss across all GPUs
                    train_perplexity += loss

        print_variable_summary()

        # calculate the mean of each gradient across all GPUs
        grads = average_gradients(tower_grads, options['batch_size'], options)
        grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
        norm_summaries.extend(norm_summary_ops)

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus)
        perplexity_summmary = tf.summary.scalar(
            'train_perplexity', train_perplexity)

        # some histogram summaries.  all models use the same parameters
        # so only need to summarize one
        histogram_summaries = [
            tf.summary.histogram('token_embedding', models[0].embedding)
        ]
        # tensors of the output from the LSTM layer
        lstm_out = tf.get_collection('lstm_output_embeddings')
        histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
        if options.get('bidirectional', False):
            # also have the backward embedding
            histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

        # apply the gradients to create the training operation
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # histograms of variables
        for v in tf.global_variables():
            histogram_summaries.append(tf.summary.histogram(v.name.replace(":", "_"), v))

        # get the gradient updates -- these aren't histograms, but we'll
        # only update them when histograms are computed
        histogram_summaries.extend(
            summary_gradient_updates(grads, opt, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge(
            [perplexity_summmary] + norm_summaries
        )
        hist_summary_op = tf.summary.merge(histogram_summaries)

        init = tf.initialize_all_variables()

    # do the training loop
    bidirectional = options.get('bidirectional', False)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(init)

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            # modified by farizikhwantri@gmail.com
            # modify here so saver will only load non softmax variables
            pretrained_var = [v for v in tf.trainable_variables() if not v.name.startswith('lm/softmax')]
            print(pretrained_var)
            loader = tf.train.Saver(pretrained_var)
            loader.restore(sess, restart_ckpt_file)
            
        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch

        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        n_train_tokens = options.get('n_train_tokens', 768648884)
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options['n_epochs'] * n_batches_per_epoch
        print("Training for %s epochs and %s batches" % (
            options['n_epochs'], n_batches_total))

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        for model in models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in options
        if char_inputs:
            max_chars = options['char_cnn']['max_characters_per_token']

        if not char_inputs:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            }
        else:
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
                for model in models
            }

        if bidirectional:
            if not char_inputs:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in models
                })
            else:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
                    for model in models
                })

        init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

        t1 = time.time()
        data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps)
        for batch_no, batch in enumerate(data_gen, start=1):

            # slice the input in the batch for the feed_dict
            X = batch
            feed_dict = {t: v for t, v in zip(
                                        init_state_tensors, init_state_values)}
            for k in range(n_gpus):
                model = models[k]
                start = k * batch_size
                end = (k + 1) * batch_size

                feed_dict.update(
                    _get_feed_dict_from_X(X, start, end, model,
                                          char_inputs, bidirectional)
                )

            # This runs the train_op, summaries and the "final_state_tensors"
            #   which just returns the tensors, passing in the initial
            #   state tensors, token ids and next token ids
            if batch_no % 1250 != 0:
                ret = sess.run(
                    [train_op, summary_op, train_perplexity] +
                                                final_state_tensors,
                    feed_dict=feed_dict
                )

                # first three entries of ret are:
                #  train_op, summary_op, train_perplexity
                # last entries are the final states -- set them to
                # init_state_values
                # for next batch
                init_state_values = ret[3:]

            else:
                # also run the histogram summaries
                ret = sess.run(
                    [train_op, summary_op, train_perplexity, hist_summary_op] + 
                                                final_state_tensors,
                    feed_dict=feed_dict
                )
                init_state_values = ret[4:]
                

            if batch_no % 1250 == 0:
                summary_writer.add_summary(ret[3], batch_no)
            if batch_no % 100 == 0:
                # write the summaries to tensorboard and display perplexity
                summary_writer.add_summary(ret[1], batch_no)
                print("Batch %s, train_perplexity=%s" % (batch_no, ret[2]))
                print("Total time: %s" % (time.time() - t1))

            if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                # save the model
                checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            if batch_no == n_batches_total:
                # done training!
                break

def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(args.vocab_file, max_word_length)

    prefix = args.train_prefix

    kwargs = {
        'test': False,
        'shuffle_on_load': True,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(prefix, vocab, **kwargs)
    else:
        data = LMDataset(prefix, vocab, **kwargs)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    # set optional inputs
    if args.n_train_tokens > 0:
        options['n_train_tokens'] = args.n_train_tokens
    if args.n_epochs > 0:
        options['n_epochs'] = args.n_epochs
    if args.batch_size > 0:
        options['batch_size'] = args.batch_size

    # train(options, data, args.n_gpus, tf_save_dir, tf_log_dir,
    #       restart_ckpt_file=ckpt_file)
    options['n_tokens_vocab'] = vocab.size
    finetune(options, data, args.n_gpus, tf_save_dir, tf_log_dir,
             restart_ckpt_file=ckpt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--n_train_tokens', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=0)

    args = parser.parse_args()
    main(args)

