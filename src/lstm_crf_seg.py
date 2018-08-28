#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/09/2018 10:23 PM
import tensorflow as tf
import tensorflow.contrib as tc
from lstm_seg import LSTMSegModel


class LSTMCRFSegModel(LSTMSegModel):

    def _output(self):
        # self.u = tc.layers.fully_connected(self.encoded_sent, self.hidden_size, tf.nn.relu, scope='output_fc1')
        # self.scores = tc.layers.fully_connected(self.u, 2, activation_fn=None, scope='output_fc2')
        self.scores = tc.layers.fully_connected(self.encoded_sent, 2, activation_fn=None, scope='output_fc1')
        self.log_likelyhood, self.trans_params = tc.crf.crf_log_likelihood(self.scores,
                                                                           tf.cast(self.placeholders['seg_labels'], tf.int32),
                                                                           self.placeholders['input_length'])

    def _compute_loss(self):
        self.loss = tf.reduce_mean(-self.log_likelyhood, 0)
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            self.loss += self.weight_decay * l2_loss

    def segment(self, batch):
        feed_dict = {self.placeholders['input_words']: batch['word_ids'],
                     self.placeholders['input_length']: batch['length'],
                     self.placeholders['dropout_keep_prob']: 1.0}

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