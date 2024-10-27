import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
import sys
from embed_data_helper import *

### Load in train and dev data for ICD top-50 and define file paths for model and save path
train_data_path = # path to MIMIC-III train_50.csv
dev_data_path = # path to MIMIC-III dev_50.csv
test_data_path = # path to MIMIC-III test_50.csv

word_model_path = # path to pre-trained word embedding model

save_train_path = # path to saved train embedding file (pt)
save_dev_path = # path to saved dev embedding file (pt)
save_test_path = # path to saved test embedding file (pt)
save_model_weight_path = # path to saved model weights (.pt)

print("Loading data:")
train_df = pd.read_csv(train_data_path)
dev_df = pd.read_csv(dev_data_path)
test_df = pd.read_csv(test_data_path)
print("Number of training notes:",len(train_df))
print("Number of validation notes:",len(dev_df))
print("Number of test notes:",len(test_df))

### Get model and define parameters
model = Word2Vec.load(word_model_path)

### Set up key pieces
num_sent = 25
sent_len = 100
max_len = 2500

dim = 100
pad = 0
padded_vec = np.zeros(dim)
padding_token = "<PAD>"
vocab = model.wv

### Get Index dictionary to get the embedding rep
word_to_index, wgt_tensor = get_word_index_wgts(model, padding_token, dim, padded_vector = padded_vec)

if wgt_tensor.shape[0] != len(vocab) + 1:
    raise AssertionError("word to index does not match size of model vocab with padding")

### Get sentence splits of train, dev and test with padding
print("Getting note splits:")
train_split = get_word_inputs(train_df, word_to_index, max_len, padding_token, sent_len)
dev_split = get_word_inputs(dev_df, word_to_index, max_len, padding_token, sent_len)
test_split = get_word_inputs(test_df, word_to_index, max_len, padding_token, sent_len)

t_size = train_split.shape
d_size = dev_split.shape
tst_size = test_split.shape

print(f'Size of training sentence split:{t_size}')
print(f'Size of dev sentence split:{d_size}')
print(f'Size of test sentence split:{tst_size}')

### Save final sentence splits:
print("Saving tensor sentence splits:")
torch.save(train_split, save_train_path)
torch.save(dev_split, save_dev_path)
torch.save(test_split, save_test_path)
torch.save(wgt_tensor, save_model_weight_path)
