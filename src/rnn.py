#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 07/25/2017 上午10:33

import tensorflow as tf
import tensorflow.contrib as tc


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
    if not rnn_type.startswith('bi'):
        cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
        state = outputs[:, -1, :]
    else:
        cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, inputs,
                                                     sequence_length=length, dtype=tf.float32)
        state_fw = outputs[0][:, -1, :]
        state_bw = outputs[1][:, -1, :]
        if concat:
            outputs = tf.concat(outputs, 2)
            state = tf.concat([state_fw, state_bw], 1)
        else:
            outputs = outputs[0] + outputs[1]
            state = state_fw + state_bw
    return outputs, state


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
    if layer_num > 1:
        cell = tc.rnn.MultiRNNCell([cell]*layer_num, state_is_tuple=True)
    return cell


