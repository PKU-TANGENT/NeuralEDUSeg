#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 03/05/2018 2:56 PM
import os
import subprocess
import random
import pickle
import logging
import spacy
from preprocess import preprocess_rst_data
from vocab import Vocab
from rst_edu_reader import RSTData
from atten_seg import AttnSegModel


def prepare(args):
    logger = logging.getLogger('SegEDU')
    logger.info('Randomly sample 10% of the training data for validation...')
    raw_train_dir = os.path.join(args.rst_dir, 'TRAINING')
    raw_dev_dir = os.path.join(args.rst_dir, 'DEV')
    if not os.path.exists(raw_dev_dir):
        os.makedirs(raw_dev_dir)
    raw_train_doc_ids = [file.split('.')[0] for file in os.listdir(raw_train_dir) if file.endswith('.out')]
    random.shuffle(raw_train_doc_ids)
    dev_doc_ids = raw_train_doc_ids[: int(len(raw_train_doc_ids) * 0.1)]
    for doc_id in dev_doc_ids:
        p = subprocess.call('mv {}/{}* {}'.format(raw_train_dir, doc_id, raw_dev_dir), shell=True)

    preprocessed_train_dir = os.path.join(args.rst_dir, 'preprocessed/train/')
    preprocessed_dev_dir = os.path.join(args.rst_dir, 'preprocessed/dev/')
    preprocessed_test_dir = os.path.join(args.rst_dir, 'preprocessed/test/')
    logger.info('Preprocessing Train data...')
    preprocess_rst_data(os.path.join(args.rst_dir, 'TRAINING'), preprocessed_train_dir)
    logger.info('Preprocessing Dev data...')
    preprocess_rst_data(os.path.join(args.rst_dir, 'DEV'), preprocessed_dev_dir)
    logger.info('Preprocessing Test data...')
    preprocess_rst_data(os.path.join(args.rst_dir, 'TEST'), preprocessed_test_dir)

    # logger.info('Building Vocab...')
    # train_files = [os.path.join(preprocessed_train_dir, filename)
    #                for filename in sorted(os.listdir(preprocessed_train_dir)) if filename.endswith('.preprocessed')]
    # dev_files = [os.path.join(preprocessed_dev_dir, filename)
    #              for filename in sorted(os.listdir(preprocessed_dev_dir)) if filename.endswith('.preprocessed')]
    # test_files = [os.path.join(preprocessed_test_dir, filename)
    #               for filename in sorted(os.listdir(preprocessed_test_dir)) if filename.endswith('.preprocessed')]
    # rst_data = RSTData(train_files=train_files, dev_files=dev_files, test_files=test_files)
    # word_vocab = Vocab(lower=False)
    # for word in rst_data.gen_all_words():
    #     word_vocab.add(word)
    #
    # logger.info('Loading pretrained embeddings for words...')
    # if args.word_embed_path:
    #     word_vocab.load_pretrained_embeddings(args.word_embed_path)
    # else:
    #     word_vocab.embed_dim = args.word_embed_size
    #
    # logger.info('Saving vocab...')
    # if not os.path.exists(os.path.dirname(args.word_vocab_path)):
    #     os.makedirs(os.path.dirname(args.word_vocab_path))
    # with open(args.word_vocab_path, 'wb') as fout:
    #     pickle.dump(word_vocab, fout)


def train(args):
    logger = logging.getLogger('SegEDU')
    logger.info('Loading data...')
    if args.train_files:
        train_files = args.train_files
    else:
        preprocessed_train_dir = os.path.join(args.rst_dir, 'preprocessed/train/')
        train_files = [os.path.join(preprocessed_train_dir, filename)
                       for filename in os.listdir(preprocessed_train_dir) if filename.endswith('.preprocessed')]
    if args.dev_files:
        dev_files = args.dev_files
    else:
        preprocessed_dev_dir = os.path.join(args.rst_dir, 'preprocessed/dev/')
        dev_files = [os.path.join(preprocessed_dev_dir, filename)
                     for filename in sorted(os.listdir(preprocessed_dev_dir)) if filename.endswith('.preprocessed')]
    if args.test_files:
        test_files = args.test_files
    else:
        preprocessed_test_dir = os.path.join(args.rst_dir, 'preprocessed/test/')
        test_files = [os.path.join(preprocessed_test_dir, filename)
                      for filename in os.listdir(preprocessed_test_dir) if filename.endswith('.preprocessed')]
    rst_data = RSTData(train_files=train_files, dev_files=dev_files, test_files=test_files)
    logger.info('Loading vocab...')
    with open(args.word_vocab_path, 'rb') as fin:
        word_vocab = pickle.load(fin)
        logger.info('Word vocab size: {}'.format(word_vocab.size()))
    rst_data.word_vocab = word_vocab
    logger.info('Initialize the model...')
    model = AttnSegModel(args, word_vocab)
    logger.info('Training the model...')
    model.train(rst_data, args.epochs, args.batch_size, print_every_n_batch=20)
    logger.info('Done with model training')


def evaluate(args):
    logger = logging.getLogger('SegEDU')
    logger.info('Loading data...')
    if args.test_files:
        test_files = args.test_files
    else:
        preprocessed_test_dir = os.path.join(args.rst_dir, 'preprocessed/test/')
        test_files = [os.path.join(preprocessed_test_dir, filename)
                      for filename in os.listdir(preprocessed_test_dir) if filename.endswith('.preprocessed')]
    rst_data = RSTData(test_files=test_files)
    logger.info('Loading vocab...')
    with open(args.word_vocab_path, 'rb') as fin:
        word_vocab = pickle.load(fin)
        logger.info('Word vocab size: {}'.format(word_vocab.size()))
    rst_data.word_vocab = word_vocab
    logger.info('Loading the model...')
    model = AttnSegModel(args, word_vocab)
    model.restore('best', args.model_dir)
    eval_batches = rst_data.gen_mini_batches(args.batch_size, test=True, shuffle=False)
    perf = model.evaluate(eval_batches, print_result=False)
    logger.info(perf)


def segment(args):
    """
    Segment raw text into edus.
    """
    logger = logging.getLogger('SegEDU')
    rst_data = RSTData()
    logger.info('Loading vocab...')
    with open(args.word_vocab_path, 'rb') as fin:
        word_vocab = pickle.load(fin)
        logger.info('Word vocab size: {}'.format(word_vocab.size()))
    rst_data.word_vocab = word_vocab
    logger.info('Loading the model...')
    model = AttnSegModel(args, word_vocab)
    model.restore('best', args.model_dir)
    if model.use_ema:
        model.sess.run(model.ema_backup_op)
        model.sess.run(model.ema_assign_op)

    spacy_nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
    for file in args.input_files:
        logger.info('Segmenting {}...'.format(file))
        raw_sents = []
        with open(file, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line:
                    raw_sents.append(line)
        samples = []
        for sent in spacy_nlp.pipe(raw_sents, batch_size=1000, n_threads=5):
            samples.append({'words': [token.text for token in sent],
                            'edu_seg_indices': []})
        rst_data.test_samples = samples
        data_batches = rst_data.gen_mini_batches(args.batch_size, test=True, shuffle=False)

        edus = []
        for batch in data_batches:
            batch_pred_segs = model.segment(batch)
            for sample, pred_segs in zip(batch['raw_data'], batch_pred_segs):
                one_edu_words = []
                for word_idx, word in enumerate(sample['words']):
                    if word_idx in pred_segs:
                        edus.append(' '.join(one_edu_words))
                        one_edu_words = []
                    one_edu_words.append(word)
                if one_edu_words:
                    edus.append(' '.join(one_edu_words))

        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        save_path = os.path.join(args.result_dir, os.path.basename(file))
        logger.info('Saving into {}'.format(save_path))
        with open(save_path, 'w') as fout:
            for edu in edus:
                fout.write(edu + '\n')
