#!/bin/bash
wget -O 'FCN_Vgg16_32s.h5' 'https://www.dropbox.com/s/27q6cmd6t4l0qsv/FCN_Vgg16_32s_epoch12.h5?dl=1'
python3 predict.py $0 $1 $2