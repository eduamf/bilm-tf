"""
ELMo usage example with pre-computed and cached context independent
token representations

"""

import sys
import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings
from smart_open import open

corpusfile = sys.argv[1]

raw_sentences = []

with open(corpusfile, 'r') as f:
    for line in f:
        res = line.strip()
        raw_sentences.append(res)

tokenized_sentences = [sentence.split() for sentence in raw_sentences]
print('We have %d sentences' % len(tokenized_sentences), file=sys.stderr)

# Create the vocabulary file with all unique tokens and
# the special <S>, </S> tokens (case sensitive).
all_tokens = {'<S>', '</S>'}
for sentence in tokenized_sentences:
    for token in sentence:
        all_tokens.add(token)

datadir = sys.argv[2]
vocab_file = os.path.join(datadir, 'temp_vocab.txt.gz')
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

# Location of pretrained LM.  Here we use the test fixtures.

options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'model.hdf5')

# Dump the token embeddings to a file. Run this once for your dataset.
print('Dumping token embeddings...', file=sys.stderr)
token_embedding_file = os.path.join(datadir, 'elmo_token_embeddings.hdf5')
dump_token_embeddings(vocab_file, options_file, weight_file, token_embedding_file)
print('Dumped token embeddings...', file=sys.stderr)
tf.reset_default_graph()

# Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

# Input placeholders to the biLM.
token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
embeddings_op = bilm(token_ids)

# elmo_input = weight_layers('input', embeddings_op, l2_coef=0.0, use_top_only=True)

# elmo_output = weight_layers('output', embeddings_op, l2_coef=0.0)

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(tokenized_sentences)


    # Compute ELMo representations
    # elmo_sentence_input_ = sess.run(elmo_input['weighted_op'], feed_dict={token_ids: sentence_ids})
    elmo_sentence_input_ = sess.run(embeddings_op['lm_embeddings'], feed_dict={token_ids: sentence_ids})
    #elmo_sentence_output_ = sess.run(elmo_output['weighted_op'],
    #                                 feed_dict={token_ids: sentence_ids})

    print(elmo_sentence_input_.shape)
    #print(elmo_sentence_output_.shape)
