#!/bin/bash

# This will run in parallel
python ~/private/HW_5/sent_split/main_run_bert.py --num_epochs 19 --batch_size 6 --bert_emb 'bluebert' --sent_split 'Yes' --dim_red 'Yes' --lr 0.001 --subset 'test' --gpu_num 4 &
python ~/private/HW_5/sent_split/main_run_bert.py --num_epochs 19 --batch_size 6 --bert_emb 'bioclinical' --sent_split 'Yes' --dim_red 'Yes' --lr 0.001 --subset 'test' --gpu_num 5 &
python ~/private/HW_5/sent_split/main_run_bert.py --num_epochs 22 --batch_size 6 --bert_emb 'roberta' --sent_split 'Yes' --dim_red 'Yes' --lr 0.001 --subset 'test' --gpu_num 6
