#!/bin/bash
# run your script
python3 main_run_bert.py --bert_emb $1 --use_label $2 --freeze_emb $3 --sentence_split $4 --dimension_reduction $5 --num_epochs $6 --lr 1e-3 --batch_size 4 --subset test
# python3 main_run_bert1.py