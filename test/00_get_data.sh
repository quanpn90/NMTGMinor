#!/usr/bin/env bash

wget http://data.statmt.org/wmt19/translation-task/dev.tgz

tar zxv --strip-components 1 -f dev.tgz \
    dev/newstest2015-ende-ref.de.sgm dev/newstest2015-ende-src.en.sgm \
    dev/newstest2016-ende-ref.de.sgm dev/newstest2016-ende-src.en.sgm \
    dev/newstest2017-deen-src.de.sgm dev/newstest2017-deen-ref.en.sgm
