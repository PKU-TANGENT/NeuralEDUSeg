#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 05/2/2018 10:10 PM

import tensorflow as tf
import tensorflow.contrib as tc


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def trilinear_similarity(x1, x2, scope='trilinear', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        x1_shape = x1.shape.as_list()
        x2_shape = x2.shape.as_list()
        if len(x1_shape) != 3 or len(x2_shape) != 3:
            raise ValueError('`args` must be 3 dims (batch_size, len, dimension)')
        if x1_shape[2] != x2_shape[2]:
            raise ValueError('the last dimension of `args` must equal')
        weights_x1 = tf.get_variable('kernel_x1', [x1_shape[2], 1], dtype=x1.dtype)
        weights_x2 = tf.get_variable('kernel_x2', [x2_shape[2], 1], dtype=x2.dtype)
        weights_mul = tf.get_variable('kernel_mul', [1, 1, x1_shape[2]], dtype=x2.dtype)
        bias = tf.get_variable('bias', [1], dtype=x1.dtype, initializer=tf.zeros_initializer)
        subres0 = tf.tile(tf.keras.backend.dot(x1, weights_x1), [1, 1, tf.shape(x2)[1]])
        subres1 = tf.tile(tf.transpose(tf.keras.backend.dot(x2, weights_x2), (0, 2, 1)), [1, tf.shape(x1)[1], 1])
        subres2 = tf.keras.backend.batch_dot(x1 * weights_mul, tf.transpose(x2, perm=(0, 2, 1)))
        return subres0 + subres1 + subres2 + tf.tile(bias, [tf.shape(x2)[1]])


def self_attention(inputs, lengths, window_size=-1, scope='bilinear_attention', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # logits = tf.matmul(inputs, inputs, transpose_b=True)  # Q * K
        logits = trilinear_similarity(inputs, inputs)
        mask = tf.sequence_mask(lengths, tf.shape(inputs)[1], tf.float32)
        mask = tf.expand_dims(mask, 1)
        if window_size > 0:
            restricted_mask = tf.matrix_band_part(tf.ones_like(logits, dtype=tf.float32), window_size, window_size)
            mask = mask * restricted_mask
        logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name='attn_weights')
        return tf.matmul(weights, inputs), weights


def rnn(rnn_type, inputs, length, hidden_size, layer_num=1,
        dropout_keep_prob=None, concat=True, scope='rnn', reuse=None):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
        scope: name scope
        reuse: reuse variables in this scope
    Returns:
        RNN outputs and final state
    """
    with tf.variable_scope(scope, reuse=reuse):
        if not rnn_type.startswith('bi'):
            cell = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=length, dtype=tf.float32)
            if rnn_type.endswith('lstm'):
                c = [state.c for state in states]
                h = [state.h for state in states]
                states = h
        else:
            cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_bw, cell_fw, inputs, sequence_length=length, dtype=tf.float32
            )
            states_fw, states_bw = states
            if rnn_type.endswith('lstm'):
                c_fw = [state_fw.c for state_fw in states_fw]
                h_fw = [state_fw.h for state_fw in states_fw]
                c_bw = [state_bw.c for state_bw in states_bw]
                h_bw = [state_bw.h for state_bw in states_bw]
                states_fw, states_bw = h_fw, h_bw
            if concat:
                outputs = tf.concat(outputs, 2)
                states = tf.concat([states_fw, states_bw], 1)
            else:
                outputs = outputs[0] + outputs[1]
                states = states_fw + states_bw
        return outputs, states


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        rnn_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        layer_num: MultiRNNCell are used if layer_num > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    cells = []
    for i in range(layer_num):
        if rnn_type.endswith('lstm'):
            # cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            cell = tc.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=hidden_size)
        elif rnn_type.endswith('gru'):
            # cell = tc.rnn.GRUCell(num_units=hidden_size)
            cell = tc.cudnn_rnn.CudnnCompatibleGRUCell(num_units=hidden_size)
        elif rnn_type.endswith('rnn'):
            cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
        else:
            raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(cell,
                                         input_keep_prob=dropout_keep_prob,
                                         output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells


