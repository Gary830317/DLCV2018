#!/bin/bash
wget -O 'seq2seq_model.h5' 'https://www.dropbox.com/s/x4ha127zcb3dpgt/seq2seq_model.h5?dl=1'
python3 output.py $0 $1 $2