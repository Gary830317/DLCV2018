#!/bin/bash
wget -O 'CNN_model.h5' 'https://www.dropbox.com/s/tb1a6invv0my25g/CNN_model.h5?dl=1'
python3 output.py $0 $1 $2 $3