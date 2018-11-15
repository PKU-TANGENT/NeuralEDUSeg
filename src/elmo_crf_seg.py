#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/09/2018 10:23 PM
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from lstm_crf_seg import LSTMCRFSegModel


class ELMOCRFSegModel(LSTMCRFSegModel):

    def __init__(self, args, word_vocab):
        super().__init__(args, word_vocab)

        # import ElmoEmbedder here so that the cuda_visible_divices can work
        from allennlp.commands.elmo import ElmoEmbedder
        self.elmo = ElmoEmbedder(cuda_device=0 if args.gpu is not None else -1)

    def _setup_placeholders(self):
        self.placeholders = {'input_words': tf.placeholder(tf.int32, shape=[None, None]),
                             'input_length': tf.placeholder(tf.int32, shape=[None]),
                             'elmo_vectors': tf.placeholder(tf.float32, shape=[None, 3, None, 1024]),
                             'seg_labels': tf.placeholder(tf.float32, shape=[None, None]),
                             'dropout_keep_prob': tf.placeholder(tf.float32)}

    def _embed(self):
        with tf.device('/cpu:0'):
            word_emb_init = tf.constant_initializer(self.word_vocab.embeddings) if self.word_vocab.embeddings is not None \
                else tf.random_normal_initializer()
            self.word_embeddings = tf.get_variable('word_embeddings',
                                                   shape=(self.word_vocab.size(), self.word_vocab.embed_dim),
                                                   initializer=word_emb_init,
                                                   trainable=False)
            self.embedded_words = tf.nn.embedding_lookup(self.word_embeddings, self.placeholders['input_words'])
        self.elmo_weights = tf.nn.softmax(tf.get_variable('elmo_weights', [3], dtype=tf.float32, trainable=True))
        self.scale_para = tf.get_variable('scale_para', [1], dtype=tf.float32, trainable=True)
        self.elmo_vectors = self.scale_para * (
            self.elmo_weights[0] * self.placeholders['elmo_vectors'][:, 0, :, :] +
            self.elmo_weights[1] * self.placeholders['elmo_vectors'][:, 1, :, :] +
            self.elmo_weights[2] * self.placeholders['elmo_vectors'][:, 2, :, :]
        )
        self.embedded_inputs = tf.concat([self.embedded_words, self.elmo_vectors], -1)
        self.embedded_inputs = tf.nn.dropout(self.embedded_inputs, self.placeholders['dropout_keep_prob'])

    def _compute_loss(self):
        self.loss = tf.reduce_mean(-self.log_likelyhood, 0)
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            self.loss += self.weight_decay * l2_loss

    def _train_epoch(self, train_batches, print_every_n_batch):
        total_loss, total_batch_num = 0, 0
        for bitx, batch in enumerate(train_batches):
            feed_dict = {self.placeholders['input_words']: batch['word_ids'],
                         self.placeholders['input_length']: batch['length'],
                         self.placeholders['seg_labels']: batch['seg_labels']}
            elmo_vectors, mask = self.elmo.batch_to_embeddings([sample['words'] for sample in batch['raw_data']])
            feed_dict[self.placeholders['elmo_vectors']] = np.asarray(elmo_vectors.data)
            feed_dict[self.placeholders['dropout_keep_prob']] = self.dropout_keep_prob

            _, loss, grad_norm = self.sess.run([self.train_op, self.loss, self.grad_norm], feed_dict)

            if bitx != 0 and print_every_n_batch > 0 and bitx % print_every_n_batch == 0:
                self.logger.info('bitx: {}, loss: {}, grad: {}'.format(bitx, loss, grad_norm))
            total_loss += loss
            total_batch_num += 1
        return total_loss / total_batch_num

    def segment(self, batch):
        feed_dict = {self.placeholders['input_words']: batch['word_ids'],
                     self.placeholders['input_length']: batch['length']}
        elmo_vectors, mask = self.elmo.batch_to_embeddings([sample['words'] for sample in batch['raw_data']])
        feed_dict[self.placeholders['elmo_vectors']] = np.asarray(elmo_vectors.data)
        feed_dict[self.placeholders['dropout_keep_prob']] = 1.0

        scores, trans_params = self.sess.run([self.scores, self.trans_params], feed_dict)

        batch_pred_segs = []
        for sample_idx in range(len(batch['raw_data'])):
            length = batch['length'][sample_idx]
            viterbi_seq, viterbi_score = tc.crf.viterbi_decode(scores[sample_idx][:length], trans_params)
            pred_segs = []
            for word_idx, label in enumerate(viterbi_seq):
                if label == 1:
                    pred_segs.append(word_idx)
            batch_pred_segs.append(pred_segs)
        return batch_pred_segs
