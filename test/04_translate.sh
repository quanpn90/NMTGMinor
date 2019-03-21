#!/usr/bin/env bash

SRC=de
TGT=en
DATADIR=data
SAVEDIR=saves

evaluate.py -load_from $1 \
            -valid_src $DATADIR/test.$SRC \
            -valid_tgt $DATADIR/test.$TGT \
            -batch_size 32 \
            $2
