#!/bin/bash
wget -O 'RNN_model.h5' 'https://www.dropbox.com/s/8reopfw7a75ng2j/RNN_model.h5?dl=1'
python3 output.py $0 $1 $2 $3