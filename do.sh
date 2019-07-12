#!/bin/bash

python train.py \
       --config=configs/m2det320_vgg.py \
       --resume_net=weights/m2det512_vgg.pth \
       --ngpu=2 \
       --tensorboard True
