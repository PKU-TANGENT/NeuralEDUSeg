#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 03/12/2018 10:10 AM
import os
import sys
import json
import spacy
import logging


logger = logging.getLogger('SegEDU')
spacy_nlp = None


def preprocess_one_doc(sent_file, edu_file):
    raw_sents = []
    with open(sent_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                raw_sents.append(line)
    raw_edus = []
    with open(edu_file, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                raw_edus.append(line)

    global spacy_nlp
    if not spacy_nlp:
        spacy_nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])

    sents = []
    for sent in spacy_nlp.pipe(raw_sents, batch_size=1000, n_threads=5):
        sents.append({'raw_text': sent.text,
                      'words': [token.text for token in sent],
                      'lemmas': [token.lemma_ for token in sent],
                      'postags': [token.pos_ for token in sent]})

    edus = []
    for edu in spacy_nlp.pipe(raw_edus, batch_size=1000, n_threads=5):
        edus.append({'raw_text': edu.text,
                     'words': [token.text for token in edu],
                     'lemmas': [token.lemma_ for token in edu],
                     'postags': [token.pos_ for token in edu]})

    cur_sent_idx = 0
    cur_sent_offset = 0
    for edu in edus:
        cur_edu_offset = 0
        while cur_edu_offset < len(edu['words']):
            if cur_sent_offset >= len(sents[cur_sent_idx]['words']):
                cur_sent_idx += 1
                cur_sent_offset = 0
            edu_word = edu['words'][cur_edu_offset]
            sent_word = sents[cur_sent_idx]['words'][cur_sent_offset]
            # print(edu_word, sent_word)
            if edu_word == sent_word:
                pass
            elif edu_word.startswith(sent_word):
                edu['words'][cur_edu_offset] = sent_word
                edu['words'].insert(cur_edu_offset + 1, edu_word[len(sent_word):])
            elif sent_word.startswith(edu_word):
                sents[cur_sent_idx]['words'][cur_sent_offset] = edu_word
                sents[cur_sent_idx]['words'].insert(cur_sent_offset + 1, sent_word[len(edu_word):])
            else:
                raise ValueError
            cur_edu_offset += 1
            cur_sent_offset += 1
        edu['sent_idx'] = cur_sent_idx
    return edus


def preprocess_rst_data(raw_data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.out'):
            logger.info('preprocessing {}...'.format(filename))
            edus = preprocess_one_doc(os.path.join(raw_data_dir, filename), os.path.join(raw_data_dir, filename + '.edus'))
            with open(os.path.join(save_dir, filename + '.preprocessed'), 'w') as fout:
                for edu in edus:
                    fout.write(json.dumps(edu) + '\n')


if __name__ == '__main__':
    preprocess_rst_data(sys.argv[1], sys.argv[2])









