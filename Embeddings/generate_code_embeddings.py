import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from embed_data_helper import *
import torch

### Load in train and dev data for ICD top-50 and define file paths for model and save path for labels
train_data_path = # path to MIMIC-III train_50.csv
dev_data_path = # path to MIMIC-III dev_50.csv
test_data_path = # path to MIMIC-III test_50.csv

code_model_path = # path to pre-trained label embedding model

save_train_path = # path to saved train embedding file (.pt)
save_dev_path = # path to saved dev embedding file (.pt)
save_test_path = # path to saved test embedding file (.pt)
save_model_weight_path = # path to saved model weights (.pt)

### read in data and model
print("Loading Data:")
train_df = pd.read_csv(train_data_path)
dev_df = pd.read_csv(dev_data_path)
test_df = pd.read_csv(test_data_path)
print("Number of training notes:",len(train_df))
print("Number of validation notes:",len(dev_df))
print("Number of test notes:",len(test_df))

## Load in model
model = Word2Vec.load(code_model_path)
ICD_codes = list(model.wv.index_to_key)
print(f'Size of label space:{len(ICD_codes)}')

### Binarize the labels for train, dev and test
train_binarize = multilabel_ohe(train_df, ICD_codes)
dev_binarize = multilabel_ohe(dev_df, ICD_codes)
test_binarize = multilabel_ohe(test_df, ICD_codes)

### Get final shape of outputs and print
t_size = train_binarize.shape
d_size = dev_binarize.shape
tst_size = test_binarize.shape

print(f'Size of training sentence split:{t_size}')
print(f'Size of dev sentence split:{d_size}')
print(f'Size of test sentence split:{tst_size}')

### Perform random check on row to ensure correct index are 1
check_multilabel(train_df, train_binarize, ICD_codes, split = "Train")
check_multilabel(dev_df, dev_binarize, ICD_codes, split = "Dev")
check_multilabel(test_df, test_binarize, ICD_codes, split = "Test")

### Additionally, get the tensor weights
code_weight_tensor = torch.tensor(model.wv.vectors)

### Save encoded label sets for data splits:
print("Saving tensor label splits:")
torch.save(train_binarize, save_train_path)
torch.save(dev_binarize, save_dev_path)
torch.save(test_binarize, save_test_path)
torch.save(code_weight_tensor, save_model_weight_path)

