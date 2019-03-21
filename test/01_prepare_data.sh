#!/usr/bin/env bash

DATADIR=data

mkdir -p $DATADIR

echo "Extracting Data"
python strip_sgml.py < newstest2015-ende-ref.de.sgm > $DATADIR/train.de
python strip_sgml.py < newstest2015-ende-src.en.sgm > $DATADIR/train.en
python strip_sgml.py < newstest2016-ende-ref.de.sgm > $DATADIR/dev.de
python strip_sgml.py < newstest2016-ende-src.en.sgm > $DATADIR/dev.en
python strip_sgml.py < newstest2017-deen-src.de.sgm > $DATADIR/test.de
python strip_sgml.py < newstest2017-deen-ref.en.sgm > $DATADIR/test.en

cd $DATADIR

echo "Applying BPE"
subword-nmt learn-joint-bpe-and-vocab --input train.de train.en -s 20000 \
            -o bpe.codes --write-vocabulary voc.de voc.en

subword-nmt apply-bpe -c bpe.codes --vocabulary voc.de --vocabulary-threshold 2 \
            < train.de > train.de.bpe
subword-nmt apply-bpe -c bpe.codes --vocabulary voc.en --vocabulary-threshold 2 \
            < train.en > train.en.bpe

subword-nmt apply-bpe -c bpe.codes --vocabulary voc.de --vocabulary-threshold 2 \
            < dev.de > dev.de.bpe
subword-nmt apply-bpe -c bpe.codes --vocabulary voc.en --vocabulary-threshold 2 \
            < dev.en > dev.en.bpe

subword-nmt apply-bpe -c bpe.codes --vocabulary voc.de --vocabulary-threshold 2 \
            < test.de > test.de.bpe
subword-nmt apply-bpe -c bpe.codes --vocabulary voc.en --vocabulary-threshold 2 \
            < test.en > test.en.bpe
