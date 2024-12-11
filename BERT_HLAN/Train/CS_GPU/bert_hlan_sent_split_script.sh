#!/bin/bash

# This will run in parallel
python ~/private/HW_5/sent_split/main_run_bert.py --num_epochs 12 --batch_size 6 --bert_emb 'bluebert' --sent_split 'Yes' --lr 0.001 --subset 'test' --gpu_num 1 &
python ~/private/HW_5/sent_split/main_run_bert.py --num_epochs 12 --batch_size 6 --bert_emb 'bioclinical' --sent_split 'Yes' --lr 0.001 --subset 'test' --gpu_num 2 &
python ~/private/HW_5/sent_split/main_run_bert.py --num_epochs 25 --batch_size 6 --bert_emb 'roberta' --sent_split 'Yes' --lr 0.001 --subset 'test' --gpu_num 3 &
