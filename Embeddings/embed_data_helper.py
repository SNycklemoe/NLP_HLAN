import numpy as np
import torch

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
    
    return split_notes