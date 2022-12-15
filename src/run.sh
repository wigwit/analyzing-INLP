#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh

conda activate news_emo

python probe.py --dataset gold --task syn --layer 8

#python eval_after_inlp.py --random --task syn