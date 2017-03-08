#!/bin/bash
set -e
python2.7 scripts/download.py
CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"
javac -cp $CLASSPATH lib/*.java
python2.7 scripts/preprocess-sst.py
python2.7 scripts/glove_compress.py
rm ./data/glove/glove.840B.300d.txt
touch ./data/glove/glove.840B.300d.txt
