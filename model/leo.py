from torch.nn import BCELoss
from sklearn.metrics import roc_auc_score
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class leo(nn.Module):
    def __init__(self, model_list, feat_dim, hidden_dim, dropout, aux_loss_weight, device):
        super(leo, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        self.leo_gnn_model = model_list[0]
        self.leo_struct_model = model_list[1]
        self.leo_proximity_model = model_list[2]
        self._loss = BCELoss()
        self.aux_loss_weight = aux_loss_weight
        
        self.ensemble_layer = leo_ensemble(feat_dim, hidden_dim, dropout, device)
        self.model_name = 'leo'

        
    def forward(self, inputs_list, edges, y=None):
        sub_scores = []
        self.auxloss = torch.zeros(1).to(self.device)

        gnn_sub_score = self.leo_gnn_model(inputs_list[0], edges)
        if not y == None:
            gnn_loss = self.leo_gnn_model._loss(gnn_sub_score, y.view(-1))
            self.auxloss += gnn_loss
        sub_scores.append(gnn_sub_score)
        embedding1 = self.leo_gnn_model.get_embedding(edges)


        struct_sub_score = self.leo_struct_model(inputs_list[1], edges)
        if not y == None:
            self.auxloss += self.leo_struct_model.auxloss
            struct_loss = self.leo_struct_model._loss(struct_sub_score, y.view(-1))
            self.auxloss += struct_loss
        sub_scores.append(struct_sub_score)
        embedding2 = self.leo_struct_model.get_embedding(edges)

        proximity_sub_score = self.leo_proximity_model(inputs_list[2], edges)
        sub_scores.append(proximity_sub_score)
        embedding3 = self.leo_proximity_model.get_embedding(edges)

        final_score = self.ensemble_layer(sub_scores, embedding1, embedding2, embedding3)
        self.auxloss *= self.aux_loss_weight
        return final_score    
            
    def auroc(self, model, inputs, loader, device):
        if loader==None: return -1
        model.eval()
        edge_index_list, y_list = [], []
        for edge_index, y in loader:
            edge_index_list.append(edge_index)
            y_list.append(y)
        edge_index = torch.cat(edge_index_list)
        y = torch.cat(y_list)
        edge_index = edge_index.to(device)
        out = model(inputs, edge_index).cpu().detach()
        return roc_auc_score(y, out)


class leo_ensemble(nn.Module):
    '''
    attention
    calculate attention weight from its edge embedding(=node embedding's max||meax)
    weight net = 2 layer / jaccard input = max||mean(dimension_reduction(node feature))
    '''
    def __init__(self, feat_dim, hidden_dim, dropout, device):
        super(leo_ensemble, self).__init__()
        self.model_name = 'leo_ensemble'
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        
        self.gnn_weight = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), self.dropout,
            nn.Linear(hidden_dim, 1), nn.Tanh(),
            )
        self.struct_weight = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), self.dropout,
            nn.Linear(hidden_dim, 1), nn.Tanh(),
            )        
        self.proximity_weight = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), self.dropout,
            nn.Linear(hidden_dim, 1), nn.Tanh(),
            )
        self.dim_reduc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.Tanh(), self.dropout,
        )
        self.device = device
        
    def forward(self, inputs, embedding1, embedding2, embedding3):
        attn_scores = []
        attn_scores.append(self.gnn_weight(self.mean_max_pool(embedding1)))

        attn_scores.append(self.struct_weight(self.mean_max_pool(embedding2)))

        embedding3 = (self.dim_reduc(embedding3[0]),self.dim_reduc(embedding3[1]))
        attn_scores.append(self.proximity_weight(self.mean_max_pool(embedding3)))

        similarities = torch.vstack(inputs).T
        attn_scores = torch.hstack(attn_scores)
        attn_scores = F.softmax(attn_scores, dim=1)
        sim = torch.mul(similarities, attn_scores).sum(dim=1)
        return self.sigmoid(sim)    
    
    def mean_max_pool(self, embeddings):
        a, b = embeddings
        mean = (a + b) / 2
        maximum = torch.maximum(a, b)
        return torch.cat([mean, maximum], dim=1)

    