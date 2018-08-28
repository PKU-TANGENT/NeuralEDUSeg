#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-2 下午3:47

import numpy as np


class Vocab(object):
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad_token, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_file(filename)

    def size(self):
        return len(self.id2token)

    def load_file(self, filename):
        for line in open(filename, 'r'):
            token = line.rstrip('\n')
            self.add(token)

    def get_id(self, key):
        key = key.lower() if self.lower else key
        if key in self.token2id:
            return self.token2id[key]
        elif key.lower() in self.token2id:
            return self.token2id[key.lower()]
        else:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        if idx in self.id2token:
            return self.id2token[idx]
        else:
            return self.unk_token

    def add(self, label, cnt=1):
        label = label.lower() if self.lower else label
        if label in self.token2id:
            idx = self.token2id[label]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = label
            self.token2id[label] = idx
        if cnt > 0:
            if label in self.token_cnt:
                self.token_cnt[label] += cnt
            else:
                self.token_cnt[label] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def load_pretrained_embeddings(self, embedding_path):
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for line in fin:
                contents = line.strip().split(' ')
                token = contents[0]
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def convert_to_ids(self, tokens):
        """Convert tokens to ids, use unk_token if the token is not in vocab."""
        vec = []
        vec += [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """Recover tokens from ids"""
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens