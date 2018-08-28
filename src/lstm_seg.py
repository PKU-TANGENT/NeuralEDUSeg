#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/08/2018 10:24 PM
import tensorflow as tf
import tensorflow.contrib as tc
from base_seg import BaseSegModel
from layers import rnn


class LSTMSegModel(BaseSegModel):
    def _build_graph(self):
        self._setup_placeholders()
        self.sequence_mask = tf.sequence_mask(self.placeholders['input_length'], dtype=tf.float32)
        with tf.variable_scope('embedding'):
            self._embed()
        with tf.variable_scope('encoding'):
            self._encode()
        with tf.variable_scope('output'):
            self._output()
        with tf.variable_scope('loss'):
            self._compute_loss()
        self.grads, self.grad_norm, self.train_op = self._get_train_op(self.loss)

    def _setup_placeholders(self):
        self.placeholders = {'input_words': tf.placeholder(tf.int32, shape=[None, None]),
                             'input_length': tf.placeholder(tf.int32, shape=[None]),
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
        self.embedded_inputs = tf.nn.dropout(self.embedded_words, self.placeholders['dropout_keep_prob'])

    def _encode(self):
        self.encoded_sent, _ = rnn('bi-lstm', self.embedded_inputs, self.placeholders['input_length'],
                                   hidden_size=self.hidden_size, layer_num=1, concat=True)
        self.encoded_sent = tf.nn.dropout(self.encoded_sent, self.placeholders['dropout_keep_prob'])

    def _output(self):
        self.u = tc.layers.fully_connected(self.encoded_sent, self.hidden_size, tf.nn.relu, scope='output_fc1')
        self.probs = tf.squeeze(tc.layers.fully_connected(self.u, 1, tf.nn.sigmoid, scope='output_fc2'), -1)

    def _compute_loss(self):
        self.losses = - tf.log(self.probs + 1e-9) * self.placeholders['seg_labels'] * self.sequence_mask \
                      - tf.log(1 - self.probs + 1e-9) * (1 - self.placeholders['seg_labels']) * self.sequence_mask
        self.loss = tf.reduce_mean(tf.reduce_sum(self.losses, 1), 0)
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            self.loss += self.weight_decay * l2_loss

    def _train_epoch(self, train_batches, print_every_n_batch):
        total_loss, total_batch_num = 0, 0
        for bitx, batch in enumerate(train_batches):
            feed_dict = {self.placeholders['input_words']: batch['word_ids'],
                         self.placeholders['input_length']: batch['length'],
                         self.placeholders['seg_labels']: batch['seg_labels'],
                         self.placeholders['dropout_keep_prob']: self.dropout_keep_prob}

            _, loss, grad_norm = self.sess.run([self.train_op, self.loss, self.grad_norm], feed_dict)

            if bitx != 0 and print_every_n_batch > 0 and bitx % print_every_n_batch == 0:
                self.logger.info('bitx: {}, loss: {}, grad: {}'.format(bitx, loss, grad_norm))
            total_loss += loss
            total_batch_num += 1
        return total_loss / total_batch_num

    def segment(self, batch):
        feed_dict = {self.placeholders['input_words']: batch['word_ids'],
                     self.placeholders['input_length']: batch['length'],
                     self.placeholders['dropout_keep_prob']: 1.0}

        batch_pred_probs = self.sess.run(self.probs, feed_dict)

        batch_pred_segs = []
        for sample_idx in range(len(batch['raw_data'])):
            pred_probs = batch_pred_probs[sample_idx]
            # self.logger.info(batch['raw_data']['words'])
            # self.logger.info(pred_probs)
            # self.logger.info(batch['raw_data']['edu_seg_indices'])
            length = batch['length'][sample_idx]
            pred_segs = []
            for word_idx, prob in enumerate(pred_probs[:length]):
                if float(prob) >= 0.4:
                    pred_segs.append(word_idx)
            batch_pred_segs.append(pred_segs)
        return batch_pred_segs
