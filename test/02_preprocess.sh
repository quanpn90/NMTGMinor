#!/usr/bin/env bash

DATADIR=data
SRC=de
TGT=en

echo "Preprocessing"
preprocess.py bilingual $DATADIR/train.$SRC.bpe $DATADIR/train.$TGT.bpe \
              -join_vocab -data_dir_out $DATADIR

preprocess.py bilingual $DATADIR/dev.$SRC.bpe $DATADIR/dev.$TGT.bpe \
              -join_vocab -data_dir_out $DATADIR
