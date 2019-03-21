#!/usr/bin/env bash

SRC=de
TGT=en

DATADIR=data
SAVEDIR=saves
LOGDIR=logs

train.py -train_src $DATADIR/train.$SRC.bpe -train_tgt $DATADIR/train.$TGT.bpe -data_dir $DATADIR \
         -src_seq_length 256 -tgt_seq_length 256 \
         -valid_src $DATADIR/dev.$SRC.bpe -valid_tgt $DATADIR/dev.$TGT.bpe \
         -model transformer \
         -layers 2 -model_size 128 \
         -batch_size_words 2000 -batch_size_update 25000 -normalize_gradient \
         -epochs 10 \
         -join_vocab -tie_weights \
         -optimizer 'adam' -learning_rate 2 -update_method 'noam' -warmup_steps 8000 \
         -seed 4535123 -log_interval 10 -save_every 10 -log_dir $LOGDIR -save_model $SAVEDIR $1
