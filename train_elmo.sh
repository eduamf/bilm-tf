#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
DATA=${1}
VOCAB=${2}
echo 'Counting tokens in the training corpora...'
SIZE=`zcat $DATA*.gz | wc -w`

echo $VOCAB
echo $SIZE

python3 bin/train_elmo.py --train_prefix ${1}  --vocab_file ${2} --save_dir ${3} --size $SIZE
