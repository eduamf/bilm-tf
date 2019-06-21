#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
DATA='PATH/*'
VOCAB='PATH.txt.gz'
SIZE=`zcat $DATA | wc -w`

echo $VOCAB
echo $SIZE

python3 bin/train_elmo.py --train_prefix "$DATA"  --vocab_file $VOCAB --save_dir ${1} --size $SIZE
