import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class HLAN_xai(nn.Module):
    def __init__(self, num_sent, word_weight_tensor, code_weight_tensor, embed_dim, hidden_size,dropout_prob,freeze_embed = True, random_word_embedding=False, use_label_embeddings=True, num_classes=50):
        super(HLAN_xai, self).__init__()
        self.num_sent = num_sent
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.use_label_embeddings = use_label_embeddings
        self.num_classes = num_classes

        if random_word_embedding:
            self.embed_layer = nn.Embedding(num_embeddings=word_weight_tensor.size(0), embedding_dim=embed_dim, padding_idx = 0)
        else:
            self.embed_layer = nn.Embedding.from_pretrained(word_weight_tensor, freeze=freeze_embed)

        self.gru_w = nn.GRU(input_size = embed_dim, hidden_size = hidden_size, bidirectional=True, batch_first=True)
        self.W_w = nn.Linear(hidden_size*2, hidden_size*2)
        self.attn_tanh = nn.Tanh()
        self.V_w_l = nn.Parameter(torch.randn(50, hidden_size*2))
        self.word_softmax = nn.Softmax(dim = 2)

        self.gru_s = nn.GRU(input_size = hidden_size * 2, hidden_size = hidden_size * 2, bidirectional = True, batch_first = True)
        self.W_s = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.V_s_l = nn.Parameter(torch.randn(50, hidden_size *2))
        self.drop_layer = nn.Dropout(p = dropout_prob)
        self.sentence_softmax = nn.Softmax(dim =2)

        if self.use_label_embeddings:
            self.W_projection = nn.Parameter(code_weight_tensor, requires_grad=not freeze_embed)
        else:
            # Initialize with Xavier (Glorot) uniform as in TensorFlow’s default initializer
            self.W_projection = nn.Parameter(torch.empty(num_classes,hidden_size * 4))
            nn.init.xavier_uniform_(self.W_projection)

        self.sigmoid_act = nn.Sigmoid()

    def forward(self, batch):
        embed_comp = self.embed_layer(batch)

        embed_comp_reshape = embed_comp.view(-1, embed_comp.shape[2], embed_comp.shape[3])

        p_attention_word, C_s_l = self.word_attention(embed_comp_reshape)

        p_attention_sent, doc_rep = self.sentence_attention(C_s_l)

        logits = self.apply_final_layer(doc_rep)

        probs = self.sigmoid_act(logits)

        return p_attention_word, p_attention_sent, probs

    def word_attention(self, X):
        hidden_state, hnn = self.gru_w(X)
        hidden_state_reshape = hidden_state.reshape(-1,hidden_state.size(-1))
        hidden_rep_step = self.attn_tanh(self.W_w(hidden_state_reshape))
        v = hidden_rep_step.reshape(-1, hidden_state.size(1),hidden_state.size(-1))

        a_w_l = torch.matmul(v,self.V_w_l.T).permute(2, 0, 1)
        p_attention_word = F.softmax(a_w_l, dim = 2)
        a_w_l = self.word_softmax(a_w_l).unsqueeze(3)
        C_s_l = a_w_l*hidden_state
        C_s_l = torch.sum(C_s_l,dim = 2)

        return p_attention_word, C_s_l

    def sentence_attention(self, X):
        X_reshape = X.transpose(0, 1)
        S_l, hnn = self.gru_s(X_reshape)

        hidden_rep_step = self.attn_tanh(self.W_s(S_l))
        hidden_rep_reshape = hidden_rep_step.view(-1, self.num_sent, hidden_rep_step.size(1), hidden_rep_step.size(2))
        U = hidden_rep_reshape.permute(2, 0, 1, 3)

        S_l_reshape = S_l.view(-1, self.num_sent, S_l.size(1), S_l.size(2))
        S_l_reshape = S_l_reshape.permute(2, 0, 1, 3)
        V_s_l_expand = self.V_s_l.unsqueeze(1).unsqueeze(1)
        attention_logits = (U * V_s_l_expand).sum(dim=3)
        p_attention_sent = self.sentence_softmax(attention_logits - attention_logits.max(dim=2, keepdim=True).values)
        document_representation = (p_attention_sent.unsqueeze(3) * S_l_reshape).sum(dim=2)
        
        return p_attention_sent, document_representation

    def apply_final_layer(self, doc_rep):
        drop_step = self.drop_layer(doc_rep) 
        drop_step_reshape = drop_step.permute(1,2,0)
        logits = drop_step_reshape * self.W_projection.T
        logits = logits.sum(dim = 1)

        return logits
    
