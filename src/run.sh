#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh

conda activate news_emo

python load_bert.py
 