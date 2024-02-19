from sklearn.metrics.pairwise import cosine_similarity

import torch.nn as nn
import numpy as np
import torch


class leo_proximity(nn.Module):
    def __init__(self, features, device, hetero=False, attack_adj=None):
        super(leo_proximity, self).__init__()
        self.model_name = 'jaccard'
        self._loss = lambda x,y: 0
        self.device = device
        if hetero:
            self.score_all = get_all_aa(attack_adj).to(device)
        else:
            self.score_all = get_all_cosine(np.array(features.cpu())).to(device)
        self.features = features
        
    def forward(self, inputs, edges):
        return self.score_all[edges[:,0], edges[:,1]]

    def get_embedding(self, edges):
        return (self.features[edges[:,0],:], self.features[edges[:,1],:])

def get_all_cosine(features):
    return torch.Tensor(cosine_similarity(features))

def get_all_aa(attack_adj):
    aa_scores = adamic_adar_fn(attack_adj)
    return torch.Tensor(aa_scores)

def adamic_adar_fn(adj):
    node_set = set()
    graph_dict = dict()
    for row ,col in zip(adj.tocoo().row, adj.tocoo().col):
        node_set.add(row)
        node_set.add(col)
        if row not in graph_dict: graph_dict[row] = set()
        graph_dict[row].add(col)
    num_nodes = adj.shape[0]
    adamic_adar = np.zeros([num_nodes, num_nodes])
    for src in node_set:
        for dst in graph_dict[src]:
            if src >= dst: continue
            aa_score = 0
            neigh_set = graph_dict[src].intersection(graph_dict[dst])

            for neigh in neigh_set:
                aa_score += 1 / len(graph_dict[neigh])
            adamic_adar[src, dst] = aa_score
            adamic_adar[dst, src] = aa_score
    return adamic_adar