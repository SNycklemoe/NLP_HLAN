import numpy as np
import torch
import random

def get_word_index_wgts(model, padding_token, dim, padded_vector):
    word_to_index = {word: idx + 1 for idx, word in enumerate(model.wv.index_to_key)}
    word_to_index[padding_token] = 0

    ## Define weight matrix based on indices
    weight_matrix = np.zeros((len(word_to_index),dim))
    for word, idx in word_to_index.items():
        if word == padding_token:
            weight_matrix[idx] = padded_vector
        else:
            weight_matrix[idx] = model.wv[word]
    weight_tensor = torch.FloatTensor(weight_matrix)

    return word_to_index, weight_tensor

def get_word_inputs(df, word_to_index, max_len, padding_token, sent_len):
    split_notes = []
    for i in range(len(df)):
        note = df['TEXT'].iloc[i]
        note_list = note.split(' ')
        note_list_pad = (note_list + max_len * [padding_token])[:max_len]

        note_indices = [word_to_index.get(token, word_to_index[padding_token]) for token in note_list_pad]

        sentences = []
        for i in range(0, max_len, sent_len):
            sentences.append(note_indices[i:i+sent_len])

        split_notes.append(sentences)
    
    return torch.LongTensor(split_notes)


def multilabel_ohe(data, codes):
    num_codes = len(codes)
    rows = []

    for i in range(len(data)):
        label = str(data['LABELS'].iloc[i]).split(';')
        indices = [i for i, value in enumerate(codes) if value in label]
        row = np.zeros(num_codes)
        row[indices] = 1
        rows.append(row)

    return torch.LongTensor(np.array(rows))

def check_multilabel(data, ohe_data, codes, split):
    rand_int = random.randint(0, len(data))
    label = str(data['LABELS'].iloc[rand_int]).split(';')
    indices = [i for i, value in enumerate(codes) if value in label]
    row = ohe_data[rand_int]

    if sum(row[indices]) != len(indices):
        raise AssertionError("Binarize multi-label does not match for random sample")
    else:
        print(f'Passed random check for {split} with rand_row:{rand_int}')
