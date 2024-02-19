from sklearn.neighbors import kneighbors_graph

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layer_1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.layer_2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.k = k
        self.relu = nn.ReLU()
        self.mlp_knn_init()
        
            
    def mlp_knn_init(self):
        self.layer_1.weight = nn.Parameter(torch.eye(self.input_dim))
        self.layer_2.weight = nn.Parameter(torch.eye(self.input_dim))
        self.layer_1.bias = nn.Parameter(torch.zeros(self.input_dim))
        self.layer_2.bias = nn.Parameter(torch.zeros(self.input_dim))
            
    def nearest_neighbors(self, X, k):
        adj = kneighbors_graph(X, k, metric='cosine')
        adj = np.array(adj.todense(), dtype=np.float32)
        adj += np.eye(adj.shape[0])
        return adj
            
    def cal_similarity_graph(self, node_embeddings):
        similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
        return similarity_graph
    
    def top_k(self, raw_graph, K):
        _, indices = raw_graph.topk(k=int(K), dim=-1)
        assert torch.max(indices) < raw_graph.shape[1]
        mask = np.zeros(raw_graph.shape)
        indices_np = indices.detach().cpu().numpy()
        mask[np.arange(raw_graph.shape[0]).reshape(-1, 1), indices_np] = 1.
        mask = torch.Tensor(mask).to(raw_graph.device)
        mask.fill_diagonal_(0) # qqq
        mask.requires_grad = False

        sparse_graph = raw_graph * mask
        return sparse_graph

    def forward(self, features):
        # 2 layer MLP
        h = self.relu(self.layer_1(features))
        embeddings = self.layer_2(h)

        # normalize한 뒤, similarity 계산
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = self.cal_similarity_graph(embeddings)
        
        # Top K 개 선택
        similarities = self.top_k(similarities, self.k + 1)
        similarities = self.relu(similarities)
        return similarities

class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output

class GCN_DAE(nn.Module):
    def __init__(self, hidden_dim, k, dropout, features_dim):
        super(GCN_DAE, self).__init__()
        # Initialize MLP based Graph Generator
        self.mlp_input_dim = features_dim
        self.mlp_hidden_dim = features_dim
        self.mlp_output_dim = features_dim
        self.k = k

        self.input_dim = features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = features_dim
        self.gcn_1 = GCNConv_dense(features_dim, self.hidden_dim)
        self.gcn_2 = GCNConv_dense(self.hidden_dim, self.output_dim)

        self.graph_gen = MLP(self.mlp_input_dim, 
                             self.mlp_output_dim, 
                             self.mlp_hidden_dim, 
                             self.k)
                
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_adj = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()   
        
    def get_adj(self, h):
        gen_adj = self.graph_gen(h)
        gen_adj =  (gen_adj + gen_adj.T) / 2  # for symmetric
        return gen_adj

    def forward(self, features, masked_features):  # x corresponds to masked_fearures
        gen_adj = self.get_adj(features)

        norm_gen_adj = gen_adj + torch.eye(len(gen_adj)).to(gen_adj.device)
        inv_sqrt_degree = 1. / (torch.sqrt(norm_gen_adj.sum(dim=1, keepdim=False)) + 1e-10)
        norm_gen_adj = inv_sqrt_degree[:, None] * norm_gen_adj * inv_sqrt_degree[None, :]

        h1 = self.relu(self.gcn_1(masked_features, norm_gen_adj))
        h2 = self.dropout(h1)
        h3 = self.gcn_2(h2, norm_gen_adj)
        return h3, gen_adj


class leo_struct_gnn(nn.Module):
    def __init__(self, dae_lp_config, slaps_lp_model, features_dim, device, binary=True, mse_flag=False):
        super(leo_struct_gnn, self).__init__()
        self.model_name = 'leo_struct_gnn'
        self.device = device
        self.mse_flag=mse_flag
        
        self.dae_model = GCN_DAE(hidden_dim=dae_lp_config['hidden_channels'],
                                 k=dae_lp_config['slaps_top_k'],
                                 dropout=dae_lp_config['dropout'],
                                 features_dim=features_dim).to(device)
        self.auxloss = 0
        self._loss = nn.BCELoss()
        self.slaps_lp_model = slaps_lp_model.to(device)
        self.mask_ratio = dae_lp_config['mask_ratio']
        self.mask_neg_ratio = dae_lp_config['mask_neg_ratio']
        self.num_epochs = dae_lp_config['dae_num_epochs']
        self.device = device
        self.binary = binary
        
    def forward(self, inputs, edges):
        features = inputs[0]
        if self.binary:
            mask = self.get_random_mask(features, self.mask_ratio, self.mask_neg_ratio)
        else:
            mask = self.get_random_mask_continuous(features, self.mask_ratio)
        masked_features = features * (1 - mask)
        recon_feat, gen_adj = self.dae_model(features, masked_features)
        indices = mask > 0
        
        if self.mse_flag:
            self.auxloss = 0.1 * F.mse_loss(recon_feat[indices], features[indices], reduction='mean')
        else:
            self.auxloss = 10 * F.binary_cross_entropy_with_logits(recon_feat[indices], features[indices], reduction='mean')
        inputs = self.lp_inputs(gen_adj, features)
        sim = self.slaps_lp_model(inputs, edges)
        self.embeddings = self.slaps_lp_model.embeddings
        return sim
        
    def get_random_mask(self, features, mask_ratio, mask_neg_ratio):
        nones = torch.sum(features > 0.0).float()
        nzeros = features.shape[0] * features.shape[1] - nones
        pzeros = (nones / nzeros) *  (mask_neg_ratio / mask_ratio)
        probs = torch.zeros(features.shape).to(features.device)
        probs[features == 0.0] = pzeros
        probs[features > 0.0] = 1 / mask_ratio
        mask = torch.bernoulli(probs)
        return mask
    
    def get_random_mask_continuous(self, features, mask_ratio):
        probs = torch.full(features.shape, 1 / mask_ratio).to(features.device)
        mask = torch.bernoulli(probs)
        return mask

    def train_dae(self, optimizer, inputs):
        features = inputs[0]
        # features = inputsf
        self.dae_model.train()

        for epoch in range(self.num_epochs):    
            mask = self.get_random_mask(features, self.mask_ratio, self.mask_neg_ratio)
            masked_features = features * (1 - mask)
            indices = mask > 0
            recon_feat, gen_adj = self.dae_model(features, masked_features)
            if self.mse_flag:
                dae_loss = 0.1 * F.mse_loss(recon_feat[indices], features[indices], reduction='mean')
            else:
                dae_loss = 10 * F.binary_cross_entropy_with_logits(recon_feat[indices], features[indices], reduction='mean')
            optimizer.zero_grad()
            loss = dae_loss
            loss.backward()
            optimizer.step()
            
    def lp_inputs(self, gen_adj, features):
        norm_gen_adj = gen_adj + torch.eye(len(gen_adj)).to(gen_adj.device)
        inv_sqrt_degree = 1. / (torch.sqrt(norm_gen_adj.sum(dim=1, keepdim=False)) + 1e-10)
        norm_gen_adj = inv_sqrt_degree[:, None] * norm_gen_adj * inv_sqrt_degree[None, :]
        inputs = [norm_gen_adj, features]            
        return inputs
    
    def get_embedding(self, edges):
        return (self.embeddings[edges[:,0],:], self.embeddings[edges[:,1],:])
