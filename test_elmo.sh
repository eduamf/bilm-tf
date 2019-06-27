#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
DATA='PATH/*'
VOCAB='PATH.txt.gz'

echo $VOCAB

python3 bin/run_test.py --test_prefix "$DATA"  --vocab_file $VOCAB --save_dir ${1}
