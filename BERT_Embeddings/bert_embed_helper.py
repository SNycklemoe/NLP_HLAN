from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

### Define function to get BERT embeddings for each word in a note
def get_bert_embeddings(input_text = None, sent_len = None, tokenizer = None, max_length_tokens = None, model = None, device = None):
    doc_rep = []
    n_notes = len(input_text)

    ### Loop over all of the notes
    for i in range(n_notes):
        ### split note based on space
        note = input_text.iloc[i].split(' ')
        note_len = len(note)
        embedding = []

        ### Loop over the words in the note, based on 100 word batches
        for j in range(0, note_len, sent_len):
            batch = note[j:j+sent_len]

            ### Get tokens for each batch, truncate based on token splits larger than max length of tokens
            encoding = tokenizer.batch_encode_plus(batch, padding = 'max_length', max_length = max_length_tokens, truncation = True,
                                                   return_tensors = 'pt', add_special_tokens = False, return_attention_mask = True)
            attention_mask = encoding['attention_mask'].to(device)

            ### Get BERT representation based on last hidden state
            with torch.no_grad():
                bert_outputs = model(**encoding.to(device))
                last_hidden = bert_outputs.last_hidden_state

            ### Get mean of BERT embeddings based on token splits. Need to account for padding based on max_length, so use attention mask
            masked_embed = last_hidden * attention_mask.unsqueeze(-1)
            non_padding = attention_mask.sum(dim = 1).unsqueeze(-1)
            embed_sum = masked_embed.sum(dim = 1)
            embed_mean = embed_sum / non_padding

            embedding.append(embed_mean.detach().cpu())
        doc_rep.append(embedding)
    return doc_rep

### Define function to pad notes from BERT embeddings to be [num_notes, num_sentences, num_words, embed_dim]
def pad_out_notes(doc_rep, sent_len, num_sent):
    padded_doc_rep = []
    n_notes = len(doc_rep)

    ### Loop over all notes
    for i in range(n_notes):
        note = doc_rep[i]
        note_len = len(note)
        loop_rep = []

        ### Loop over all sentences in the notes
        for j in range(note_len):
            sent_rep = note[j]
            ### If sentence is not as long as sentence length, pad with 0s
            if sent_rep.shape[0] != sent_len:
                padded_note = F.pad(sent_rep, pad = (0, 0, 0, sent_len - sent_rep.shape[0]))
                loop_rep.append(padded_note)
            else:
                loop_rep.append(sent_rep)
        
        ### For note, add additional sentences with 0s to ensure there are 25 sentences
        for k in range(num_sent - note_len):
            loop_rep.append(torch.zeros(sent_len, 768))

        loop_rep = torch.stack(loop_rep)
        padded_doc_rep.append(loop_rep)
    ### Stack appended notes to ensure tensor is returned
    padded_doc_rep_tensor = torch.stack(padded_doc_rep)

    return padded_doc_rep_tensor