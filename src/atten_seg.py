#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 04/09/2018 10:23 PM
import numpy as np
import tensorflow as tf
from rnn import rnn
from elmo_crf_seg import ELMOCRFSegModel
from layers import self_attention


class AttnSegModel(ELMOCRFSegModel):

    def _encode(self):
        with tf.variable_scope('rnn_1'):
            self.encoded_sent, _ = rnn('bi-lstm', self.embedded_inputs, self.placeholders['input_length'],
                                       hidden_size=self.hidden_size, layer_num=1, concat=True)
            self.encoded_sent = tf.nn.dropout(self.encoded_sent, self.placeholders['dropout_keep_prob'])
        self.attn_outputs, self.attn_weights = self_attention(
            self.encoded_sent, self.placeholders['input_length'], self.window_size)
        self.attn_outputs = tf.nn.dropout(self.attn_outputs, self.placeholders['dropout_keep_prob'])
        self.encoded_sent = tf.concat([self.encoded_sent, self.attn_outputs], -1)
        with tf.variable_scope('rnn_2'):
            self.encoded_sent, _ = rnn('bi-lstm',
                                       self.encoded_sent,
                                       self.placeholders['input_length'],
                                       hidden_size=self.hidden_size, layer_num=1, concat=True)
            self.encoded_sent = tf.nn.dropout(self.encoded_sent, self.placeholders['dropout_keep_prob'])
