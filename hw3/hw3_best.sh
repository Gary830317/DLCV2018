#!/bin/bash
wget -O 'VggUnet.h5' 'https://www.dropbox.com/s/1brs1cwhuhoflrg/VGGUnet_epoch34_saved.h5?dl=1'
python3 predict.py $0 $1 $2