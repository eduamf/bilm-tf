# python3
# coding: utf-8

import sys
import csv
import numpy as np
from gensim import models
import zipfile
import logging
import json
import re
import os
from smart_open import open
import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from gensim.matutils import unitvec


def tokenize(string):
    token_pattern = re.compile('(?u)\w+')
    tokens = [t.lower() for t in token_pattern.findall(string)]
    return tokens


def load_dataset(data_file, max_tokens=350):
    lens = []
    data = csv.reader(open(data_file), delimiter='\t')
    _ = next(data)
    data_set = {}
    cur_lemma = None
    word_set = []
    for row in data:
        i, lemma, sense_id, left, word, right, senses = row
        if lemma != cur_lemma:
            if len(word_set) > 0:
                data_set[cur_lemma] = word_set
            word_set = []
            cur_lemma = lemma
        sent = ' '.join([left, word, right])
        tok_sent = tokenize(sent)
        tok_sent = tok_sent[:max_tokens]  # We keep only the first tokens, to reduce batch size
        sent = ' '.join(tok_sent)
        sent_len = len(tok_sent)
        lens.append(sent_len)
        # if len(tokenize(sent)) > 400:
        #    print(sent_len)
        #    print(word)
        #    print(sent)
        cl = int(sense_id)
        num = len(tokenize(left))
        word_set.append((sent, num, cl))
    data_set[cur_lemma] = word_set
    print('Dataset loaded')
    print('Sentences:', len(lens))
    print('Max length:', np.max(lens))
    print('Average length:', np.average(lens))
    print('Standard deviation:', np.std(lens))
    return data_set


def get_dummy_vector():
    vect = np.random.rand(1, 100)
    return vect[0]


def get_word_vector(text, model, num):
    """
        :param text: list of words
        :param model: word2vec model in Gensim format
        :param num: number of the word to exclude
        :return: average vector of words in text
        """
    # Creating list of all words in the document which are present in the model
    excl_word = text[num]
    words = [w for w in text if w in model and w != excl_word]
    lexicon = list(set(words))
    lw = len(lexicon)
    if lw < 1:
        print('Empty lexicon in', text, file=sys.stderr)
        return np.zeros(model.vector_size)
    vectors = np.zeros((lw, model.vector_size))  # Creating empty matrix of vectors for words
    for i in list(range(lw)):  # Iterate over words in the text
        word = lexicon[i]
        vectors[i, :] = model[word]  # Adding word and its vector to matrix
    semantic_fingerprint = np.sum(vectors, axis=0)  # Computing sum of all vectors in the document
    semantic_fingerprint = np.divide(semantic_fingerprint, lw)  # Computing average vector
    return semantic_fingerprint


def get_elmo_vector(sess, texts, batcher, sentence_character_ids, elmo_sentence_input, nrs):
    vectors = []

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print('Sentences in this chunk:', len(texts), file=sys.stderr)

    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})
    print('ELMo sentence input shape:', elmo_sentence_input_.shape, file=sys.stderr)

    for sentence, nr in zip(range(len(texts)), nrs):
        # query_word = texts[sentence][nr]
        # print(texts[sentence])
        query_vec = elmo_sentence_input_[sentence, nr, :]
        query_vec = unitvec(query_vec)
        # print('Vector shape:', query_vec.shape)
        vectors.append(query_vec)
    return vectors

def get_elmo_vector_average(sess, texts, batcher, sentence_character_ids, elmo_sentence_input, nrs):
    vectors = []

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(texts)
    print('Sentences in this chunk:', len(texts), file=sys.stderr)

    # Compute ELMo representations.
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})
    print('ELMo sentence input shape:', elmo_sentence_input_.shape, file=sys.stderr)

    for sentence in range(len(texts)):
        sent_vec = np.zeros((elmo_sentence_input_.shape[1], elmo_sentence_input_.shape[2]))
        for word_vec in enumerate(elmo_sentence_input_[sentence, :, :]):
            sent_vec[word_vec[0], :] = word_vec[1]
        semantic_fingerprint = np.sum(sent_vec, axis=0)
        semantic_fingerprint = np.divide(semantic_fingerprint, sent_vec.shape[0])
        query_vec = unitvec(semantic_fingerprint)
        vectors.append(query_vec)
    return vectors


def load_word2vec_embeddings(embeddings_file):
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    # Detect the model format by its extension:
    # Binary word2vec format:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        emb_model = models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True,
                                                             unicode_errors='replace')
    # Text word2vec format:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        emb_model = models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    # ZIP archive from the NLPL vector repository:
    elif embeddings_file.endswith('.zip'):
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open('meta.json')
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print('============')
            # Loading the model itself:
            stream = archive.open("model.bin")  # or model.txt, if you want to look at the model
            emb_model = models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:
        # Native Gensim format?
        emb_model = models.KeyedVectors.load(embeddings_file)
        # If you intend to train further: emb_model = models.Word2Vec.load(embeddings_file)

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return emb_model


def load_elmo_embeddings(directory):
    vocab_file = os.path.join(directory, 'vocab.txt.gz')
    options_file = os.path.join(directory, 'options.json')
    weight_file = os.path.join(directory, 'model.hdf5')

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, 50)

    # Input placeholders to the biLM.
    sentence_character_ids = tf.placeholder('int32', shape=(None, None, 50))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=300)

    # Get ops to compute the LM embeddings.
    sentence_embeddings_op = bilm(sentence_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    # Our model includes ELMo at both the input and output layers
    # of the task GRU, so we need 2x ELMo representations at each of the input and output.

    elmo_sentence_input = weight_layers('input', sentence_embeddings_op, use_top_only=True)
    return batcher, sentence_character_ids, elmo_sentence_input


def divide_chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i:i + n]
