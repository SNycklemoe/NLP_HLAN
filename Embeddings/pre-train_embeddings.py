## Python file to train the label (code) and word based embeddings for HLAN with word2vec
### ICD-9 label sets from MIM-III used for training with splits from CAML-MIMIC-III

## Import libraries
import pandas as pd
from gensim.models import Word2Vec
import os

## Designate number of cores for works
num_cores = os.cpu_count()
print(f'Number of cores for model pre-training: {num_cores - 1}')

## Designate file paths for the input data and output model
### Label
#label_train_path = # path to MIMIC_3 train_full.csv
label_train_path = # path to MIMIC_3 train_50.csv (reduce label space)
label_model_path = # path to save label embedding model

### word
word_train_path = # path to MIMIC_3 disch_full.csv
word_model_path = # path to save word embedding model

### Label Based embeddings ###
data = pd.read_csv(label_train_path)

## Get labels
label_set = []
for i in range(len(data)):
    label_set.append(data['LABELS'].iloc[i])
label_set = [str(label).split(';') for label in label_set]

## Train Word2Vec
print("Training label based word2vec:")
dim = 400
model = Word2Vec(label_set, vector_size = dim, window = 5, min_count = 0, workers = num_cores - 1)

if len(model.wv) != 50:
    raise AssertionError('Error: Label set size is not 50')

## Save model:
print("Saving label word2vec model at:",label_model_path)
model.save(label_model_path)

### Word Based embeddings ###
data = pd.read_csv(word_train_path)

## Get words:
word_set = []
for i in range(len(data)):
    word_set.append(data['TEXT'].iloc[i])
word_set = [word.split(' ') for word in word_set]

## Train Word2Vec
print("Training word based word2vec:")
dim = 100
model = Word2Vec(word_set, vector_size = dim, window = 5, min_count = 0, workers = num_cores - 1)

## Save model:
print("Saving word word2vec model at:",word_model_path)
model.save(word_model_path)