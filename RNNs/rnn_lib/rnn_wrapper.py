import torch
import torch.nn.functional as F

class RNNCellWrapper:
    def __init__(self, rnn_cell, nonlinearity='relu'):
        self.rnn_cell = rnn_cell
        self.activation = nonlinearity
        self.n_hid = rnn_cell.n_hidden
        self.U = rnn_cell.U
        self.W = rnn_cell.W
        self.b = rnn_cell.b

    def forward(self, x, h):
        """x: (batch, input_dim), h: (batch, hidden_dim)"""
        pre_activation = torch.matmul(x, self.U.T) + torch.matmul(h, self.W.T) + self.b.T
        if self.activation == 'relu':
            h = F.relu(pre_activation)
        else:
            h = pre_activation

        return h
    
    
    def eval(self):  # <- just a stub
        pass

    @property
    def nonlinearity(self):
        return self.activation

    @property
    def n_hidden(self):
        return self.n_hid

