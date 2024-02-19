from .gcn import GraphConvolution
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
   
class leo_gnn(nn.Module):
    def __init__(self, num_features, hidden_channels, dropout):
        super(leo_gnn, self).__init__()
        self.model_name = 'leo_gnn'
        self._loss = BCELoss()
        self.gc1 = GraphConvolution(num_features, hidden_channels)
        self.gc2 = GraphConvolution(hidden_channels, hidden_channels)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        
        self.weight_lin = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.bias_lin = nn.Parameter(torch.Tensor(hidden_channels))
        nn.init.xavier_uniform_(self.weight_lin)
        nn.init.zeros_(self.bias_lin)
        
        self._loss = nn.BCELoss()
        self.auxloss = 0
        
    def forward(self, inputs, edges):
        norm_adj, norm_features = inputs
        x = self.tanh(self.gc1(norm_features, norm_adj))
        x = self.dropout(x)
        
        x = self.tanh(self.gc2(x, norm_adj)) 
        embeddings = self.dropout(x)

        self.embeddings = embeddings
        sym_weight = (self.weight_lin + self.weight_lin.T) / 2
        sim = (embeddings[edges[:,0],:] @ sym_weight) * embeddings[edges[:,1],:] + self.bias_lin
        sim = sim.sum(dim=1)
        return self.sigmoid(sim)    
    
    def get_embedding(self, edges):
        return (self.embeddings[edges[:,0],:], self.embeddings[edges[:,1],:])

    