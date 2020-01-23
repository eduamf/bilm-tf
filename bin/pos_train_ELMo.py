import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import scipy.spatial.distance as ds
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# python bin/dump_weights.py --save_dir swb/checkpoint --outfile swb/model/manga_weights.hdf5

# Incluído para limpar o escopo
tf.reset_default_graph()

# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('swb', 'model')
vocab_file = os.path.join(datadir, 'vocab_manga.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'manga_weights.hdf5')
 
# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)
 
# Input placeholders to the biLM.
context_character_ids = tf.compat.v1.placeholder('int32', shape=(None, None, 50))
 
# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)
 
# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_character_ids)
 
# Get an op to compute ELMo (weighted average of the internal biLM layers)
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
 
# Now we can compute embeddings.
raw_context = ['manga longas começam do ombro até o pulso .',
               'usar mangas curtas em climas quentes é mais agradável .',
               'a manga é fruta de clima tropical .',
               'o sumo da manga é delicioso .'] 
 
tokenized_context = [sentence.split() for sentence in raw_context]
print(tokenized_context)

##################################
# Imprimir as sentenças por token
##################################

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())
 
    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    print("Shape of context ids = ", context_ids.shape)
 
    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_ = sess.run(
        elmo_context_input['weighted_op'],
        feed_dict={context_character_ids: context_ids}
    )
 
print("Shape of generated embeddings = ",elmo_context_input_.shape)

#####################################

# [['manga', 'longas', 'começam', 'do', 'ombro', 'até', 'o', 'pulso', '.'],
# 0 0
#  ['usar', 'mangas', 'curtas', 'em', 'climas', 'quentes', 'é', 'mais', 'agradável', '.'],
# 1 1
#  ['a', 'manga', 'é', 'de', 'clima', 'tropical', '.'],
# 2 1
#  ['o', 'sumo', 'da', 'manga', 'é', 'delicioso', '.']]
# 3 3

# Computing euclidean distance between words embedding
print("Euclidean Distance Comparison Manga - ")

euc_dist_manga_s0w0_s1w1 = np.linalg.norm(elmo_context_input_[0,0,:]
                                        - elmo_context_input_[1,1,:])
print("\nSentence 1 Word 1 x Sentence 2 Word 2 = ", 
      np.round(euc_dist_manga_s0w0_s1w1, 2))

euc_dist_manga_s0w0_s2w1 = np.linalg.norm(elmo_context_input_[0,0,:]
                                        - elmo_context_input_[2,1,:])
print("\nSentence 1 Word 1 x Sentence 3 Word 2 = ", 
      np.round(euc_dist_manga_s0w0_s2w1, 2))

euc_dist_manga_s0w0_s3w3 = np.linalg.norm(elmo_context_input_[0,0,:]
                                        - elmo_context_input_[3,3,:])
print("\nSentence 1 Word 1 x Sentence 4 Word 4 = ", 
      np.round(euc_dist_manga_s0w0_s3w3, 2))

euc_dist_manga_s2w1_s3w3 = np.linalg.norm(elmo_context_input_[2,1,:]
                                        - elmo_context_input_[3,3,:])
print("\nSentence 3 Word 2 x Sentence 4 Word 4 = ", 
      np.round(euc_dist_manga_s2w1_s3w3, 2))

# Computing cosine distance between words embedding
print("\n\nCosine Distance Comparison Manga - ")

cos_dist_manga_s0w0_s1w1 = ds.cosine(elmo_context_input_[0,0,:]
                                    ,elmo_context_input_[1,1,:])
print("\nSentence 1 Word 1 x Sentence 2 Word 2 = "
      , np.round(cos_dist_manga_s0w0_s1w1, 3))

cos_dist_manga_s0w0_s2w1 = ds.cosine(elmo_context_input_[0,0,:]
                                    ,elmo_context_input_[2,1,:])
print("\nSentence 1 Word 1 x Sentence 3 Word 2 = "
      , np.round(cos_dist_manga_s0w0_s2w1, 3))

cos_dist_manga_s0w0_s3w3 = ds.cosine(elmo_context_input_[0,0,:]
                                    ,elmo_context_input_[3,3,:])
print("\nSentence 1 Word 1 x Sentence 4 Word 4 = "
      , np.round(cos_dist_manga_s0w0_s3w3, 3))

cos_dist_manga_s2w1_s3w3 = ds.cosine(elmo_context_input_[2,1,:]
                                    ,elmo_context_input_[3,3,:])
print("\nSentence 3 Word 2 x Sentence 4 Word 4 = "
      , np.round(cos_dist_manga_s2w1_s3w3, 3))




     