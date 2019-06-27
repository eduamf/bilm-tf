#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
#DATA='data/iphuck/*'
#DATA='data/rnc_texts/*'
DATA='data/rnc_tokens_test/*'
#VOCAB='data/rnc_vocab_reduced.txt.gz'
VOCAB='data/rnc_tokens_vocab.txt.gz'

echo $VOCAB


# We pass as arguments the path to the training corpus and (optionally) the number of CPU cores we want to use
/opt/conda/envs/python3.6/bin/python3 bin/run_test.py --test_prefix "$DATA"  --vocab_file $VOCAB --save_dir ${1}
