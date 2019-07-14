#!/bin/bash

python eval_model.py \
       --config=configs/m2det320_vgg.py \
       --trained_model=weights/Final_M2Det_COCO_size320_netvgg16.pth \
       --output=output.json
