import torch
import torch.nn as nn
from torch.autograd import Variable

# https://github.com/hunkim/PyTorchZeroToAll/blob/master/13_2_rnn_classification.py
# - written with this, use the one below
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

# TRY THIS - THIS ONE WORKED!!!
# https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L15/1_lstm.ipynb

###########################################################################
################################# RNN #####################################
###########################################################################

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, n_layers = 1, dropout = 0, bias = False):
        super(RNNClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        # self.embedding = (input_size, hidden_size)
        
        # input_size might need to be hidden_size as well
        #nonlinearity='relu',
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first = True, bias = bias)
        self.drop = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, sequences):

        batch_size = sequences.size(0)
    
        #embedded = self.embedding(sequence)
#         print(sequences.shape)

        # hidden = self._init_hidden(batch_size)
        
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).float().to(self.device)
        
        out, hidden = self.rnn(sequences, hidden) # embedded here for sequence if not commented out
                
        out = self.drop(hidden[-1])  
        out = self.linear(out)
    
        return out, hidden
    
    # def _init_hidden(self, batch_size):
        #  hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
    #     return Variable(hidden)

###########################################################################
################################ LSTM #####################################
###########################################################################

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, n_layers = 1, dropout = 0, bias = False):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        # self.embedding = (input_size, hidden_size)
        
        # input_size might need to be hidden_size as well
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first = True, bias = bias)
        self.drop = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, sequences):

        batch_size = sequences.size(0)
    
        #embedded = self.embedding(sequence)
#         print(sequences.shape)

        # hidden = self._init_hidden(batch_size)
        # hidden = hidden.to(self.device)
        
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).float().to(self.device)
        
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size).float().to(self.device)
        out, (hidden, cell) = self.lstm(sequences, (hidden, cell)) # embedded here for sequence if not commented out
    
#         output, hidden = self.lstm(sequences)
        out = self.drop(hidden[-1])  
        out = self.linear(out)
    
        return out, hidden
    
    # def _init_hidden(self, batch_size):
    #     hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
    #     return Variable(hidden)
    
    
###########################################################################
################################# GRU #####################################
###########################################################################

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, n_layers = 1, dropout = 0, bias = False):
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        # self.embedding = (input_size, hidden_size)
        
        # input_size might need to be hidden_size as well
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first = True, bias = bias)
        self.drop = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, sequences):

        batch_size = sequences.size(0)
    
        #embedded = self.embedding(sequence)
#         print(sequences.shape)

        # hidden = self._init_hidden(batch_size)
        # hidden = hidden.to(self.device)
        
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).float().to(self.device)
        
        out, hidden = self.gru(sequences, hidden) # embedded here for sequence if not commented out
        out = self.drop(hidden[-1])  
        out = self.linear(out)
    
        return out, hidden
    
    # def _init_hidden(self, batch_size):
    #     hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
    #     return Variable(hidden)