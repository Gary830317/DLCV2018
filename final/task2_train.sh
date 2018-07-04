#!/bin/bash
wget -O 'data_augmentation_Resnet.h5' 'https://www.dropbox.com/s/ojbd8z2db3txgbr/data_augmentation_Resnet.h5?dl=1'
python3 train.py $1 $2 $3 $4

