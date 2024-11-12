from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from bert_embed_helper import *
from datetime import datetime

### Load in train and dev data for ICD top-50 and define save paths
train_data_path = "~/train_50.csv" # path to train_50.csv
dev_data_path = "~/dev_50.csv" # path to dev_50.csv
test_data_path = "~/test_50.csv" # path to test_50.csv

save_train_path = "~/train_word_embed_roberta.pt" # path to save roberta train embeddings
save_dev_path = "~/dev_word_embed_roberta.pt" # path to save roberta dev embeddings
save_test_path = "~/test_word_embed_roberta.pt" # path to save roberta test embeddings

print("Loading data:")
train_df = pd.read_csv(train_data_path)
dev_df = pd.read_csv(dev_data_path)
test_df = pd.read_csv(test_data_path)
print("Number of training notes:",len(train_df))
print("Number of validation notes:",len(dev_df))
print("Number of test notes:",len(test_df))

### Load in BERT models from Transformers Library
roberta_tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
roberta_model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

### Set up GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Set up hyperparameters
num_sent = 25
sent_len = 100
max_len = 2500
max_length_tokens = 25

### Get Embeddings
train_text = train_df['TEXT']
dev_text = dev_df['TEXT']
test_text = test_df['TEXT']
roberta_model.to(device)

print("Started Embedding Generation: at time:",datetime.now())
train_bert_embed = get_bert_embeddings(input_text = train_text, sent_len = sent_len, tokenizer = roberta_tokenizer, max_length_tokens = max_length_tokens, model = roberta_model, device = device)
train_embeddings = pad_out_notes(doc_rep = train_bert_embed, sent_len = sent_len, num_sent = num_sent)

dev_bert_embed = get_bert_embeddings(input_text = dev_text, sent_len = sent_len, tokenizer = roberta_tokenizer, max_length_tokens = max_length_tokens, model = roberta_model, device = device)
dev_embeddings = pad_out_notes(doc_rep = dev_bert_embed, sent_len = sent_len, num_sent = num_sent)

test_bert_embed = get_bert_embeddings(input_text = test_text, sent_len = sent_len, tokenizer = roberta_tokenizer, max_length_tokens = max_length_tokens, model = roberta_model, device = device)
test_embeddings = pad_out_notes(doc_rep = test_bert_embed, sent_len = sent_len, num_sent = num_sent)
print("Finished Embedding Generation at time:",datetime.now())

print(f'Shape of Train embeddings:{train_embeddings.shape}')
print("Saving Training embeddings at:", save_train_path)
torch.save(train_embeddings, save_train_path)

print(f'Shape of Dev embeddings:{dev_embeddings.shape}')
print("Saving Dev embeddings at:", save_dev_path)
torch.save(dev_embeddings, save_dev_path)

print(f'Shape of Test embeddings:{test_embeddings.shape}')
print("Saving Test embeddings at:", save_test_path)
torch.save(test_embeddings, save_test_path)