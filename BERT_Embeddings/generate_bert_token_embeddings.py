from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from bert_embed_helper import *
from datetime import datetime

### Define file paths for input data and for saving the tokenized input and corresponding attention masks
train_data_path = #train_50.csv
dev_data_path = #dev_50.csv
test_data_path = #test_50.csv"

save_train_token_path = #"~/train_tok_bert_embed"
save_train_mask_path = #"~/train_mask_bert_embed"
save_dev_token_path = #"~/dev_tok_bert_embed"
save_dev_mask_path = #"~/dev_mask_bert_embed"
save_test_token_path = #"~/test_tok_bert_embed"
save_test_mask_path = #"~/test_mask_bert_embed"

### Load in data
print("Loading data:")
train_df = pd.read_csv(train_data_path)
dev_df = pd.read_csv(dev_data_path)
test_df = pd.read_csv(test_data_path)
print("Number of training notes:",len(train_df))
print("Number of validation notes:",len(dev_df))
print("Number of test notes:",len(test_df))

### Load in BERT tokenizers from Transformers Library
bluebert_tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
bioclinical_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
roberta_tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

### Set up hyperparameters
num_sent = 100
sent_len = 25
max_len = 2500
max_length_tokens = 64

### Embedding Loop
train_text = train_df['TEXT']
dev_text = dev_df['TEXT']
test_text = test_df['TEXT']

tokenizers = [bluebert_tokenizer, bioclinical_tokenizer, roberta_tokenizer]
model_paths = ["_bluebert.pt","_bioclinical.pt","_roberta.pt"]
train_tokens, train_mask = tokenize_bert_and_pad(input_text = train_text, sent_len = sent_len, num_sent = num_sent, tokenizer = bioclinical_tokenizer, max_length_tokens = max_length_tokens)


print("Started Tokenizing at time:",datetime.now())
for i in range(len(tokenizers)):
    print(f'Tokenizing: Loop {i}')

    train_tokens, train_mask = tokenize_bert_and_pad(input_text = train_text, sent_len = sent_len, num_sent = num_sent, tokenizer = tokenizers[i], max_length_tokens = max_length_tokens)
    print(f'Shape of Train tokenized:{train_tokens.shape}')
    print("Saving Training tokenized at:", save_train_token_path + model_paths[i])
    torch.save(train_tokens, save_train_token_path + model_paths[i])
    print("Saving Training attention mask at:", save_train_mask_path + model_paths[i])
    torch.save(train_mask, save_train_mask_path + model_paths[i])

    dev_tokens, dev_mask = tokenize_bert_and_pad(input_text = dev_text, sent_len = sent_len, num_sent = num_sent, tokenizer = tokenizers[i], max_length_tokens = max_length_tokens)
    print(f'Shape of Dev tokenized:{dev_tokens.shape}')
    print("Saving Dev tokenized at:", save_dev_token_path + model_paths[i])
    torch.save(dev_tokens, save_dev_token_path + model_paths[i])
    print("Saving Dev attention mask at:", save_dev_mask_path + model_paths[i])
    torch.save(dev_mask, save_dev_mask_path + model_paths[i])

    test_tokens, test_mask = tokenize_bert_and_pad(input_text = test_text, sent_len = sent_len, num_sent = num_sent, tokenizer = tokenizers[i], max_length_tokens = max_length_tokens)
    print(f'Shape of Test tokenized:{test_tokens.shape}')
    print("Saving Test tokenized at:", save_test_token_path + model_paths[i])
    torch.save(test_tokens, save_test_token_path + model_paths[i])
    print("Saving Test attention mask at:", save_test_mask_path + model_paths[i])
    torch.save(test_mask, save_test_mask_path + model_paths[i])

    print(f'End Tokenizing: Loop {i}')
print("Finished Tokenizing at time:",datetime.now())
