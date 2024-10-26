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