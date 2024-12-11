import torch
import numpy as np

def get_rows_true_positive_top_ten(x, y, batch_size, model, top_ten_ind, calibration):
    true_pos_top_ten_ind = []
    for i in range(0,len(x), batch_size):
        x_comp = x[i:i+batch_size, : ,:]
        y_comp = y[i:i+batch_size, :]
        model.eval()
        with torch.no_grad():
            probs = model(x_comp)[2]
        for j in range(len(x_comp)):
            labels = np.where(y_comp[j])
            preds = np.where(probs[j] > calibration)
            true_pos = np.intersect1d(labels[0], preds[0])
            true_pos_top = np.intersect1d(true_pos, top_ten_ind)
            if len(true_pos_top) != 0:
                true_pos_top_ten_ind.append(i + j)
    return true_pos_top_ten_ind

def get_label_space_example(ex_ind, x, y, model, top_ten_df,calibration):
    x_comp = x[ex_ind:ex_ind+1, :, :]
    y_comp = y[ex_ind:ex_ind+1, :]
    model.eval()
    with torch.no_grad():
        p_attention_word, p_attention_sent, probs = model(x_comp)
    labels = np.where(y_comp)
    preds = np.where(probs > calibration)
    true_pos = np.intersect1d(labels[1], preds[1])
    top_ten_match = []
    top_ten_match_label = []
    attn_sent_match = []
    attn_word_wgt_match = []
    for label in true_pos:
        if label in top_ten_df['Index'].values:
            top_ten_match.append(label)
            top_ten_match_label.append(top_ten_df[top_ten_df['Index'] == label]['Label'])
            p_attn_sent = p_attention_sent[label,:]
            p_attn_word = p_attention_word[label, :, :]
            attn_word_wgt = p_attn_word * p_attn_sent.T
            attn_sent_match.append(p_attn_sent)
            attn_word_wgt_match.append(attn_word_wgt)
    
    return top_ten_match, top_ten_match_label, attn_sent_match, attn_word_wgt_match

def get_examples_with_wgts_hlan(num_examples, ex_ind,label, x, attn_sent, attn_word, word_model, window_size = 3):
    x_comp = x[ex_ind:ex_ind+1,:,:]
    num_words = attn_word.shape[1]
    top_5_wgt_words = np.argsort(attn_word[0], axis = None)[-num_examples:]
    row_idx, col_idx = np.unravel_index(top_5_wgt_words, attn_word.shape)
    print(f'example for:{label}','\n')
    for i in range(num_examples -1, -1, -1):
        sent_ind = row_idx[i]
        word_ind = col_idx[i]
        word_start = max(0, word_ind - window_size)
        word_end = min(num_words, word_ind + window_size + 1)
        example_sent = x_comp[0, sent_ind, word_start:word_end]
        ex_word_ind = x_comp[0,sent_ind, word_ind]
        ex_list = []
        for idx in example_sent:
            word_from_index = word_model.wv.index_to_key[idx]
            ex_list.append(word_from_index)
        ex_word = word_model.wv.index_to_key[ex_word_ind.item()]
        print('Example sentence:',ex_list)
        print(f'Sentence weight:{attn_sent[:,sent_ind].item()}')
        print(f'Wgt Word Attn for {ex_word}: {attn_word[sent_ind,word_ind]}','\n')

def get_seaborn_hlan(ex_ind, x, attn_word):
    x_comp = x[ex_ind:ex_ind+1,:,:]
    mask_sum = np.sum(np.array(x_comp[0]), axis = 1)
    mask_rows = np.where(mask_sum != 0)[0]
    attn_mat = np.array(attn_word[mask_rows])
    return attn_mat


def get_examples_with_wgts_bert(num_examples, ex_ind, label, x, x_mask, bert_model,bert_tokenizer, window_size = 3):
    x_comp = x[ex_ind:ex_ind+1,:,:]
    x_comp_m = x_mask[ex_ind: ex_ind + 1, :]
    bert_model.eval()
    with torch.no_grad():
        p_attention_word, p_attn_sent, probs = bert_model(x_comp, x_comp_m)
    attention_word = p_attention_word[label.index[0],:,:] * x_comp_m.squeeze(0)
    attn_sent = p_attn_sent[label.index[0],:]
    attn_word = attention_word * attn_sent.T
    num_words = attn_word.shape[1]
    top_5_wgt_words = np.argsort(attn_word[0], axis = None)[-num_examples:]
    row_idx, col_idx = np.unravel_index(top_5_wgt_words, attn_word.shape)
    print(f'example for:{label}','\n')
    for i in range(num_examples -1, -1, -1):
        sent_ind = row_idx[i]
        word_ind = col_idx[i]
        word_start = max(0, word_ind - window_size)
        word_end = min(num_words, word_ind + window_size)
        example_sent = x_comp[0, sent_ind, word_start:word_end]
        ex_word_ind = x_comp[0,sent_ind, word_ind]
        ex_list = []
        for idx in example_sent:
            word_from_index = bert_tokenizer.convert_ids_to_tokens(idx.item())
            if word_from_index[0] == 'Ġ':
                word_from_index = word_from_index[1:]
            ex_list.append(word_from_index)
        ## post processing the hashes:
        ex_list_cleaned = []
        for i in range(len(ex_list)):
            word = ex_list[i]
            if word[0] == '#':
                word = word.lstrip('#')
                if i == 0:
                    ex_list_cleaned.append(word)
                else:
                    ex_list_cleaned[-1] = ex_list_cleaned[-1] + word
            else:
                ex_list_cleaned.append(word)

        ex_word = bert_tokenizer.convert_ids_to_tokens(ex_word_ind.item())
        if ex_word[0] == 'Ġ':
            ex_word = ex_word[1:]
        ex_word_wgt = attn_word[sent_ind,word_ind]
        if ex_word[0] == '#':
            ex_word = ex_word.lstrip('#')
            for i in range(word_ind, word_ind + 3):
                word_add_ind= x_comp[0, sent_ind, i + 1]
                word_add = bert_tokenizer.convert_ids_to_tokens(word_add_ind.item())
                if word_add[0] == '#':
                    ex_word_wgt += attn_word[sent_ind, i + 1]
                    word_add = word_add.lstrip('#')
                    ex_word = ex_word + word_add
                else:
                    break
            for i in range(word_ind - 1, word_ind - 3, -1):
                word_add_ind = x_comp[0, sent_ind, i]
                word_add = bert_tokenizer.convert_ids_to_tokens(word_add_ind.item())
                if word_add[0] == '#':
                    ex_word_wgt += attn_word[sent_ind, i]
                    word_add = word_add.lstrip('#')
                    ex_word = word_add + ex_word
                else:
                    ex_word_wgt += attn_word[sent_ind, i]
                    word_add = word_add.lstrip('#')
                    ex_word = word_add + ex_word
                    break
        print('Example sentence cleaned:',ex_list_cleaned)
        print(f'Sentence weight:{attn_sent[:,sent_ind].item()}')
        print(f'Wgt Word Attn for {ex_word}: {ex_word_wgt}','\n')


def get_seaborn_bert(ex_ind, label, x, x_mask, bert_model):
    x_comp = x[ex_ind:ex_ind+1,:,:]
    x_comp_m = x_mask[ex_ind: ex_ind + 1, :]
    bert_model.eval()
    with torch.no_grad():
        p_attention_word, p_attn_sent, probs = bert_model(x_comp, x_comp_m)
    attention_word = p_attention_word[label.index[0],:,:]
    attn_sent = p_attn_sent[label.index[0],:]
    attn_word = attention_word * attn_sent.T

    mask_sum = np.sum(np.array(x_comp[0]), axis = 1)
    mask_rows = np.where(mask_sum != 0)[0]
    attn_mat = np.array(attn_word[mask_rows])

    return attn_mat