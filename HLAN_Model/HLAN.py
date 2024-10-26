import torch
import numpy as np



class HLAN():
    def __init__(self) -> None:
        pass
    def gru_single_step_word_level(self, Xt, h_t_minus_1):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size*num_sentences,embed_size]
        :param h_t_minus_1:[batch_size*num_sentences,embed_size]
        :return:
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = torch.sigmoid(Xt @ self.W_z + h_t_minus_1 @ self.U_z + self.b_z) # z_t:[batch_size*num_sentences,self.hidden_size]
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = torch.sigmoid(Xt @ self.W_r + h_t_minus_1 @ self.U_r + self.b_r) # r_t:[batch_size*num_sentences,self.hidden_size]
        # candiate state h_t~
        h_t_candidate = torch.tanh(Xt @ self.W_h + r_t * (h_t_minus_1 @ self.U_h) + self.b_h) # h_t_candiate:[batch_size*num_sentences,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candidate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t
    
    def gru_single_step_sentence_level(self, Xt, h_t_minus_1):  # Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        # update gate: decides how much past information is kept and how much new information is added.
        z_t = torch.sigmoid(Xt @ self.W_z_sentence + h_t_minus_1 @ self.U_z_sentence +self.b_z_sentence)
        # z_t:[batch_size,self.hidden_size*2]
        #print('z_t in gru_single_step_sentence_level', z_t.get_shape()) # z_t in gru_single_step_sentence_level (128, 200)                                                             
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = torch.sigmoid(Xt @ self.W_r_sentence + h_t_minus_1 @ self.U_r_sentence + self.b_r_sentence)  
        # r_t:[batch_size,self.hidden_size*2]
        #print('r_t in gru_single_step_sentence_level', r_t.get_shape()) # r_t in gru_single_step_sentence_level (128, 200)                                                                             
        # candiate state h_t~
        h_t_candidate = torch.tanh(Xt @ self.W_h_sentence + r_t * (h_t_minus_1 @ self.U_h_sentence) + self.b_h_sentence)
        # h_t_candiate:[batch_size,self.hidden_size*2]
        #print('h_t_candiate in gru_single_step_sentence_level', h_t_candiate.get_shape()) # h_t_candiate in gru_single_step_sentence_level (128, 200)    
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candidate
        #print('h_t in gru_single_step_sentence_level', h_t.get_shape()) # h_t in gru_single_step_sentence_level (128, 200)            
        return h_t
    
    # GRU single step for sentence level per label
    def gru_single_step_sentence_level_per_label(self, Xt,
                                       h_t_minus_1):  # Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
        """
        single step of gru for sentence level
        :param Xt:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
        :param h_t_minus_1:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
        :return:h_t:[batch_size,hidden_size]
        """
        #expand dimension to the same rank for tf.matmul
        W_z_sentence = self.W_z_sentence.unsqueeze(0).expand(self.num_classes, -1, -1)  # Expand to [num_classes, hidden_size*2, hidden_size*2]
        U_z_sentence = self.U_z_sentence.unsqueeze(0).expand(self.num_classes, -1, -1)
        W_r_sentence = self.W_r_sentence.unsqueeze(0).expand(self.num_classes, -1, -1)
        U_r_sentence = self.U_r_sentence.unsqueeze(0).expand(self.num_classes, -1, -1)
        W_h_sentence = self.W_h_sentence.unsqueeze(0).expand(self.num_classes, -1, -1)
        U_h_sentence = self.U_h_sentence.unsqueeze(0).expand(self.num_classes, -1, -1)

        # update gate: decides how much past information is kept and how much new information is added.
        z_t = z_t = torch.sigmoid(Xt @ W_z_sentence + h_t_minus_1 @ U_z_sentence + self.b_z_sentence) 
        # z_t:[batch_size,self.hidden_size*2]
        #print('z_t in gru_single_step_sentence_level', z_t.get_shape()) # z_t in gru_single_step_sentence_level (128, 200)                                                             
        # reset gate: controls how much the past state contributes to the candidate state.
        r_t = torch.sigmoid(Xt @ W_r_sentence + h_t_minus_1 @ U_r_sentence + self.b_r_sentence)  
        # r_t:[batch_size,self.hidden_size*2]
        #print('r_t in gru_single_step_sentence_level', r_t.get_shape()) # r_t in gru_single_step_sentence_level (128, 200)                                                                             
        # candiate state h_t~
        h_t_candidate = torch.tanh(Xt @ W_h_sentence + r_t * (h_t_minus_1 @ U_h_sentence) + self.b_h_sentence)
        # h_t_candiate:[batch_size,self.hidden_size*2]
        #print('h_t_candiate in gru_single_step_sentence_level', h_t_candiate.get_shape()) # h_t_candiate in gru_single_step_sentence_level (128, 200)    
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candidate 
        # this is element-wise multiplication
        #print('h_t in gru_single_step_sentence_level', h_t.get_shape()) # h_t in gru_single_step_sentence_level (128, 200)            
        return h_t
    
    def gru_forward_word_level(self, embedded_words):
        """
        :param embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return:forward hidden state: a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = torch.split(embedded_words, self.sequence_length, dim=1) 
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        # Now the sequence_length is the sentence_length
        #print('after splitting in gru', len(embedded_words_splitted), embedded_words_splitted[0].get_shape())       
        embedded_words_squeeze = [x.squeeze(1) for x in embedded_words_splitted]
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=embedded_words_squeeze[0].get_shape().dims[0]
        #h_t = tf.ones((self.batch_size * self.num_sentences,
        #               self.hidden_size))  
        # #TODO self.hidden_size h_t =int(tf.get_shape(embedded_words_squeeze[0])[0]) # tf.ones([self.batch_size*self.num_sentences, self.hidden_size]) # [batch_size*num_sentences,embed_size]
        h_t = torch.ones_like(embedded_words_squeeze[0])
        h_t_forward_list = []
        for Xt in embedded_words_squeeze:
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_forward_list.append(h_t)
        # Xt: [batch_size*num_sentences,embed_size]
        # [batch_size*num_sentences,embed_size]<------Xt:[batch_size*num_sentences,embed_size];h_t:[batch_size*num_sentences,embed_size]
        return h_t_forward_list  # a list,length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
    

    def gru_backward_word_level(self, embedded_words):
        """
        :param   embedded_words:[batch_size*num_sentences,sentence_length,embed_size]
        :return: backward hidden state:a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embedded_words
        embedded_words_splitted = torch.split(embedded_words,self.sequence_length, dim = 1)
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,1,embed_size]
        embedded_words_squeeze = [x.squeeze(1) for x in embedded_words_splitted]
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        embedded_words_squeeze.reverse() 
        # it is a list,length is sentence_length, each element is [batch_size*num_sentences,embed_size]
        # demension_1=int(tf.get_shape(embedded_words_squeeze[0])[0]) #h_t = tf.ones([self.batch_size*self.num_sentences, self.hidden_size])
        #h_t = tf.ones((self.batch_size * self.num_sentences, self.hidden_size))
        h_t = torch.ones_like(embedded_words_squeeze[0])
        h_t_backward_list = []
        for Xt in embedded_words_squeeze:
            h_t = self.gru_single_step_word_level(Xt, h_t)
            h_t_backward_list.append(h_t)
        h_t_backward_list.reverse() #ADD 2017.06.14
        return h_t_backward_list
    
    def gru_forward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size*2]
        """
        # split embedded_words
        sentence_representation_splitted = torch.split(sentence_representation,self.num_sentences,dim = 1)
        # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [x.squeeze(1) for x in sentence_representation_splitted]
        # it is a list.length is num_sentences,each element is [batch_size, hidden_size*2]
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0]) #scalar: batch_size
        #h_t = tf.ones((self.batch_size, self.hidden_size * 2))  # TODO
        h_t = torch.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for Xt in sentence_representation_squeeze:  
            # Xt:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt,h_t)  
            # h_t:[batch_size,hidden_size*2]<---------Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is num_sentences, each element is [batch_size,hidden_size*2]
    
    def gru_forward_sentence_level_per_label(self, sentence_representation):
        """
        :param sentence_representation: [num_classes,batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [num_classes,batch_size,hidden_size*2]
        """
        # split embedded_words
        sentence_representation_splitted = torch.split(sentence_representation, self.num_sentences,dim=2)  
        # it is a list.length is num_sentences,each element is  [num_classes,batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [x.squeeze(2) for x in sentence_representation_splitted]  
        # it is a list.length is num_sentences,each element is [num_classes,batch_size, hidden_size*2]
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0]) #scalar: batch_size
        #h_t = tf.ones((self.num_classes, self.batch_size, self.hidden_size * 2))
        h_t = torch.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for Xt in sentence_representation_squeeze:  
            # Xt:[num_classes, batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level_per_label(Xt,h_t)  
            # h_t:[num_classes,batch_size,hidden_size*2]<---------Xt:[num_classes, batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2] or [num_classes, batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        return h_t_forward_list  # a list,length is num_sentences, each element is [num_classes, batch_size,hidden_size*2]
    
    def gru_backward_sentence_level(self, sentence_representation):
        """
        :param sentence_representation: [batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = torch.split(sentence_representation, self.num_sentences, dim=1)  
        # it is a list.length is num_sentences,each element is [batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [x.squeeze(1) for x in sentence_representation_splitted]  
        # it is a list.length is num_sentences,each element is [batch_size, hidden_size*2]
        sentence_representation_squeeze.reverse()
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0])  # scalar: batch_size
        #h_t = tf.ones((self.batch_size, self.hidden_size * 2))
        h_t = torch.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for Xt in sentence_representation_squeeze:  
            # Xt:[batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level(Xt,h_t)  
            # h_t:[batch_size,hidden_size*2]<---------Xt:[batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse() #ADD 2017.06.14
        return h_t_forward_list  # a list,length is num_sentences, each element is [batch_size,hidden_size*2]
    
    def gru_backward_sentence_level_per_label(self, sentence_representation):
        """
        :param sentence_representation: [num_classes, batch_size,num_sentences,hidden_size*2]
        :return:forward hidden state: a list,length is num_sentences, each element is [num_classes,batch_size,hidden_size]
        """
        # split embedded_words
        sentence_representation_splitted = torch.split(sentence_representation, self.num_sentences,dim=2)  
        # it is a list.length is num_sentences,each element is [num_classes,batch_size,1,hidden_size*2]
        sentence_representation_squeeze = [x.squeeze(2) for x in sentence_representation_splitted]  
        # it is a list.length is num_sentences,each element is [num_classes,batch_size, hidden_size*2]
        sentence_representation_squeeze.reverse()
        # demension_1 = int(tf.get_shape(sentence_representation_squeeze[0])[0])  # scalar: batch_size
        #h_t = tf.ones((self.num_classes, self.batch_size, self.hidden_size * 2))
        h_t = torch.ones_like(sentence_representation_squeeze[0])
        h_t_forward_list = []
        for Xt in sentence_representation_squeeze:  
            # Xt:[num_classes, batch_size, hidden_size*2]
            h_t = self.gru_single_step_sentence_level_per_label(Xt,h_t)  
            # h_t:[num_classes,batch_size,hidden_size*2]<---------Xt:[num_classes,batch_size, hidden_size*2]; h_t:[batch_size, hidden_size*2]
            h_t_forward_list.append(h_t)
        h_t_forward_list.reverse() #ADD 2017.06.14
        return h_t_forward_list  # a list,length is num_sentences, each element is [num_classes,batch_size,hidden_size*2]











