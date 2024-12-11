#!/bin/bash
# run your script
python3 main_run.py --use_label $1 --freeze_emb $2 --random_word_emb $3 --num_epochs $4 --lr 1e-4 --subset test

