#!/bin/bash
wget -O '1_shot_Resnet.h5' 'https://www.dropbox.com/s/q4ahm4hjapxmuuj/1_shot_Resnet.h5?dl=1'
wget -O '5_shot_Resnet.h5' 'https://www.dropbox.com/s/2azu0kiz2uczd3f/5_shot_Resnet.h5?dl=1'
wget -O '10_shot_Resnet.h5' 'https://www.dropbox.com/s/26r91parmuylviw/10_shot_Resnet.h5?dl=1'
python3 output.py $1 $2 $3