# python3
# coding: utf-8

import sys
import csv
import numpy as np
from gensim import models
import zipfile
import logging
import json


def load_dataset(data_file):
    data = csv.reader(open(data_file), delimiter='\t')
    _ = next(data)
    data_set = {}
    cur_lemma = None
    word_set = []
    for row in data:
        i, lemma, sense_id, left, word, right, senses = row
        if lemma != cur_lemma:
            cur_lemma = lemma
            if len(word_set) > 0:
                data_set[cur_lemma] = word_set
            word_set = []
        sent = ' '.join([left, word, right])
        cl = int(sense_id)
        num = len(left.split(' '))
        word_set.append((sent, num, cl))
    return data_set


def get_dummy_vector():
    vect = np.random.rand(1, 1024)
    return vect[0]


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


def get_word_vector(text, model, num):
    """
        :param text: list of words
        :param model: word2vec model in Gensim format
        :param num: number of the word to exclude
        :return: average vector of words in text
        """
    # Creating list of all words in the document, which are present in the model
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
