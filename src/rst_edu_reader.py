#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 03/05/2018 2:56 PM
import json
import random
import numpy as np


class RSTData(object):
    def __init__(self, train_files=None, dev_files=None, test_files=None):
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files
        self.train_samples = self.read_samples(train_files) if train_files else []
        self.dev_samples = self.read_samples(dev_files) if dev_files else []
        self.test_samples = self.read_samples(test_files) if test_files else []
        self.word_vocab = None

    def read_samples(self, files):
        samples = []
        for file in files:
            with open(file, 'r') as fin:
                sent_edus = {}
                for line in fin:
                    edu_info = json.loads(line)
                    if edu_info['sent_idx'] in sent_edus:
                        sent_edus[edu_info['sent_idx']].append(edu_info)
                    else:
                        sent_edus[edu_info['sent_idx']] = [edu_info]
                for sent_idx in sorted(sent_edus.keys()):
                    words, edu_seg_indices = [], []
                    for edu_info in sent_edus[sent_idx]:
                        for word in edu_info['words']:
                            words.append(word)
                        if len(edu_seg_indices) == 0:
                            edu_seg_indices.append(len(edu_info['words']))
                        else:
                            edu_seg_indices.append(edu_seg_indices[-1] + len(edu_info['words']))
                    edu_seg_indices.pop(-1)
                    samples.append({'words': words, 'edu_seg_indices': edu_seg_indices})
        return samples

    def gen_all_words(self):
        for sample in self.train_samples + self.dev_samples + self.test_samples:
            for word in sample['words']:
                yield word

    def gen_all_postags(self):
        for sample in self.train_samples + self.dev_samples + self.test_samples:
            for postag in sample['postags']:
                yield postag

    def gen_mini_batches(self, batch_size, train=False, dev=False, test=False, shuffle=False):
        samples = []
        if train:
            samples += self.train_samples
        if dev:
            samples += self.dev_samples
        if test:
            samples += self.test_samples
        data_size = len(samples)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self.one_mini_batch(samples, batch_indices)

    def one_mini_batch(self, samples, batch_indices):
        raw_data = [samples[i] for i in batch_indices]
        batch_data = {'raw_data': raw_data,
                      'word_ids': [],
                      'length': [], 'seg_labels': []}
        for sidx, sample in enumerate(batch_data['raw_data']):
            word_ids = self.word_vocab.convert_to_ids(sample['words'])
            seg_labels = [1 if index in sample['edu_seg_indices'] else 0 for index in range(len(word_ids))]
            batch_data['word_ids'].append(word_ids)
            batch_data['length'].append(len(word_ids))
            batch_data['seg_labels'].append(seg_labels)
        batch_data, padded_len = self.dynamic_padding(batch_data)
        return batch_data

    def dynamic_padding(self, batch_data):
        max_len = max(batch_data['length'])
        word_pad_id = self.word_vocab.get_id(self.word_vocab.pad_token)
        batch_data['word_ids'] = [(ids + [word_pad_id] * (max_len - len(ids)))[: max_len]
                                  for ids in batch_data['word_ids']]
        batch_data['seg_labels'] = [(labels + [0] * (max_len - len(labels)))[: max_len]
                                    for labels in batch_data['seg_labels']]
        return batch_data, max_len

