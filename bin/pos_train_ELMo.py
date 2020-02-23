import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '10'

import tensorflow as tf
tf.compat.v1.logging.info('TensorFlow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.logging.info('TensorFlow')

import numpy as np
import scipy.spatial.distance as ds
import pandas as pd
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# python bin/dump_weights.py --save_dir swb/checkpoint --outfile swb/model/manga_weights.hdf5

Teste = 3

# Incluído para limpar o escopo
tf.reset_default_graph()

# Location of pretrained LM.  Here we use the test fixtures.
modeldir = os.path.join('swb', 'model')

vocab_file = os.path.join(modeldir, 'manga_vocab.txt' if (Teste==1 or Teste==3) else 'ombra_vocab.txt')
options_file = os.path.join(modeldir, 'manga_options.json' if (Teste==1 or Teste==3) else 'ombra_options.json')
weight_file = os.path.join(modeldir, 'manga_weights.hdf5' if (Teste==1 or Teste==3) else 'ombra_weights.hdf5')
 
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

# Now we can compute embeddings for some sents.
raw_context1 = ['manga longas começam do ombro até o pulso .',
               'manga curtas são mais apertadas em camisa e em vestido .',
               'a manga tem polpa doce e suculenta .',
               'o sumo da manga é muito apreciado , uma delícia .'] 

raw_context2 = ['ombra longas começam do ombro até o pulso .', 
                'ombra curtas são mais apertadas em camisa e em vestido .', 
                'a manga tem polpa doce e suculenta .',
                'o sumo da manga é muito apreciado , uma delícia .']

if Teste==1:
    raw_context = raw_context1 
elif Teste==2:
    raw_context = raw_context2
else:
    # C:\Projetos\ELMo\bilm-tf\bin\swb\train\mangas
    traindir = os.path.join('swb', 'train', 'mangas', 'pre_Manga_mix4.txt')
    with open(traindir, "r", encoding = "utf-8") as f:
        lines = f.readlines()
    raw_context = [manga_sent for manga_sent in lines if 'manga' in manga_sent]

tokenized_context = [sentence.split() for sentence in raw_context]

##################################
# Imprimir as sentenças por token
##################################
if (Teste==1 or Teste==2):
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

    print(tokenized_context)
    #########################################################################################
    # Computing euclidean distance between words embedding
    print("Euclidean Distance Comparison Manga Fruit (1 & 2) and Manga Vestment (3 & 4)) - T", Teste )
    # 1
    euc_dist_manga_s0_s1 = np.linalg.norm(elmo_context_input_[0,0,:]
                                            - elmo_context_input_[1,0,:])
    print("\nSentence 1 x Sentence 2 = ", tokenized_context[0][0],
          np.round(euc_dist_manga_s0_s1, 2), tokenized_context[1][0])
    euc_dist_manga_s0_s2 = np.linalg.norm(elmo_context_input_[0,0,:]
                                            - elmo_context_input_[2,1,:])
    print("\nSentence 1 x Sentence 3 = ", tokenized_context[0][0], 
          np.round(euc_dist_manga_s0_s2, 2), tokenized_context[2][1])
    euc_dist_manga_s0_s3 = np.linalg.norm(elmo_context_input_[0,0,:]
                                            - elmo_context_input_[3,3,:])
    print("\nSentence 1 x Sentence 4 = ", tokenized_context[0][0], 
          np.round(euc_dist_manga_s0_s3, 2), tokenized_context[3][3],)
    # 2
    euc_dist_manga_s1_s2 = np.linalg.norm(elmo_context_input_[1,0,:]
                                            - elmo_context_input_[2,1,:])
    print("\nSentence 2 x Sentence 3 = ", tokenized_context[1][0], 
          np.round(euc_dist_manga_s1_s2, 2), tokenized_context[2][1],)
    euc_dist_manga_s1_s3 = np.linalg.norm(elmo_context_input_[1,0,:]
                                            - elmo_context_input_[3,3,:])
    print("\nSentence 2 x Sentence 4 = ", tokenized_context[1][0], 
          np.round(euc_dist_manga_s1_s3, 2), tokenized_context[3][3],)
    # 3
    euc_dist_manga_s2_s3 = np.linalg.norm(elmo_context_input_[2,1,:]
                                            - elmo_context_input_[3,3,:])
    print("\nSentence 3 x Sentence 4 = ", tokenized_context[2][1], 
          np.round(euc_dist_manga_s2_s3, 2), tokenized_context[3][3],)
    # Computing cosine distance between words embedding
    print("\n\nCosine Distance Comparison Manga (Fruit) and Manga (Clothing) - T", Teste)
    # 1
    cos_dist_manga_s0_s1 = ds.cosine(elmo_context_input_[0,0,:]
                                        ,elmo_context_input_[1,0,:])
    print("\nSentence 1 x Sentence 2 = ", tokenized_context[0][0]
          , np.round(cos_dist_manga_s0_s1, 3), tokenized_context[1][0])
    cos_dist_manga_s0_s2 = ds.cosine(elmo_context_input_[0,0,:]
                                        ,elmo_context_input_[2,1,:])
    print("\nSentence 1 x Sentence 3 = ", tokenized_context[0][0]
          , np.round(cos_dist_manga_s0_s2, 3), tokenized_context[2][1])
    cos_dist_manga_s0_s3 = ds.cosine(elmo_context_input_[0,0,:]
                                        ,elmo_context_input_[3,3,:])
    print("\nSentence 1 x Sentence 4 = ", tokenized_context[0][0]
          , np.round(cos_dist_manga_s0_s3, 3), tokenized_context[3][3])
    # 2
    cos_dist_manga_s0w0_s2w0 = ds.cosine(elmo_context_input_[1,0,:]
                                        ,elmo_context_input_[2,1,:])
    print("\nSentence 2 x Sentence 3 = ", tokenized_context[1][0]
          , np.round(cos_dist_manga_s0_s2, 3), tokenized_context[2][1])
    cos_dist_manga_s1_s3 = ds.cosine(elmo_context_input_[1,0,:]
                                        ,elmo_context_input_[3,3,:])
    print("\nSentence 2 x Sentence 4 = ", tokenized_context[1][0]
          , np.round(cos_dist_manga_s1_s3, 3), tokenized_context[3][3])
    # 3
    cos_dist_manga_s2_s3 = ds.cosine(elmo_context_input_[2,1,:]
                                        ,elmo_context_input_[3,3,:])
    print("\nSentence 3 x Sentence 4 = ", tokenized_context[2][1]
          , np.round(cos_dist_manga_s2_s3, 3), tokenized_context[3][3])
    ######################################################################
else:
    lbl = open('manga_vec_lbl.tsv', "w", encoding = "utf-8")
    prime = True
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
     
        # Create loop of data (memory constraint).
        size = len(tokenized_context)
        valores = []
        for i in range(size):
            if 'manga' in tokenized_context[i]:
                manga_pos = tokenized_context[i].index("manga")
                if prime:
                    linha = " ".join(tokenized_context[i])
                    prime = False
                else: 
                    linha = "\n" + " ".join(tokenized_context[i])
            lbl.write(linha)
            tokens = [tokenized_context[i]]
            context_ids = batcher.batch_sentences(tokens)
            # Compute ELMo representations (here for the input only, for simplicity).
            elmo_context_input_ = sess.run(
                elmo_context_input['weighted_op'],
                feed_dict={context_character_ids: context_ids}
            )
            vetor = elmo_context_input_[0,manga_pos,:]
            inorm = 1 / np.linalg.norm(vetor)
            valores.append(vetor * inorm)
            #norm = np.linalg.norm(valores)
        # salva
        df_manga = pd.DataFrame(valores)
        df_manga.to_csv('manga_vec.tsv', sep = '\t', header=False, index=None)


