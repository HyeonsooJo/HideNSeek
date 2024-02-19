import torch
import numpy as np
import scipy.sparse as sp
import random
import os 
import pdb
from sklearn.metrics import roc_auc_score

from .unnot_utils import *


def get_all_jaccard(features):
    # features numpy array
    intersection = (features @ features.T)

    features_col = features.sum(axis=1, keepdims=True)
    features_col = np.repeat(features_col, features.shape[0], axis=1)
    features_row = features.sum(axis=1, keepdims=True).T
    features_row = np.repeat(features_row, features.shape[0], axis=0)
    jaccard = intersection / (features_col + features_row - intersection + 1e-12)
    jaccard = torch.Tensor(jaccard)
    return jaccard

def jaccard_fn(features, edge_index):    
    y_preds = []    
    for src, dst in edge_index.T:
        feat_src = features[src]
        feat_dst = features[dst]
        intersection = np.count_nonzero(feat_src * feat_dst)
        jaccard =  intersection / (np.count_nonzero(feat_src) + np.count_nonzero(feat_dst) - intersection+1e-23)
        y_preds.append(jaccard)
    return torch.tensor(y_preds)

def svd_fn(features, edge_index, path_attacked_adj, attack_adj=None):
    if attack_adj==None:
        attack_adj = torch.load(path_attacked_adj)
    rank = 100
    U, S, V = np.linalg.svd(attack_adj.toarray())
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    diag_S = np.diag(S)
    denoise_adj = U @ diag_S @ V
    y_preds = denoise_adj[edge_index[0,:], edge_index[1,:]]
    return torch.tensor(y_preds)

def cosine_fn(features, edge_index):    
    y_preds = []    
    for src, dst in edge_index.T:
        feat_src = features[src]
        feat_dst = features[dst]
        cosine = np.dot(feat_src, feat_dst)/(np.linalg.norm(feat_src)*np.linalg.norm(feat_dst))
        if np.isnan(cosine): cosine=0
        y_preds.append(cosine)
    return torch.tensor(y_preds)

def svd_score(ori_adj, attack_adj, features):
    rank = 5
    U, S, V = np.linalg.svd(attack_adj.toarray())
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    diag_S = np.diag(S)
    denoise_adj = U @ diag_S @ V

    attack_edge_index = np.stack(((attack_adj - ori_adj)>0).nonzero())
    attack_edge_index = attack_edge_index[:, attack_edge_index[0, :] < attack_edge_index[1, :]]

    real_edge_index = np.stack(ori_adj.nonzero())
    real_edge_index = real_edge_index[:, real_edge_index[0, :] < real_edge_index[1, :]]
    edge_index = np.hstack((real_edge_index, attack_edge_index))
    y_preds = denoise_adj[edge_index[0,:], edge_index[1,:]]
    y_trues = torch.LongTensor([1] * real_edge_index.shape[1] + [0] * attack_edge_index.shape[1])
    return roc_auc_score(y_trues, y_preds)   

def adamic_adar_fn(adj):
    node_set = set()
    graph_dict = dict()
    for row ,col in zip(adj.tocoo().row, adj.tocoo().col):
        if row == col:
            print(row, col)
            print("ERROR")
            exit()
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

def adamic_score(ori_adj, attack_adj, features):
    aa_scores = adamic_adar_fn(attack_adj)
    attack_edge_index = np.stack(((attack_adj - ori_adj)>0).nonzero())
    attack_edge_index = attack_edge_index[:, attack_edge_index[0, :] < attack_edge_index[1, :]]

    real_edge_index = np.stack(ori_adj.nonzero())
    real_edge_index = real_edge_index[:, real_edge_index[0, :] < real_edge_index[1, :]]
    edge_index = np.hstack((real_edge_index, attack_edge_index))
    y_preds = aa_scores[edge_index[0,:], edge_index[1,:]]
    y_trues = torch.LongTensor([1] * real_edge_index.shape[1] + [0] * attack_edge_index.shape[1])
    return roc_auc_score(y_trues, y_preds)   

def cosine_score(ori_adj, attack_adj, features):

    attack_edge_index = np.stack(((attack_adj - ori_adj)>0).nonzero())
    attack_edge_index = attack_edge_index[:, attack_edge_index[0, :] < attack_edge_index[1, :]]
    attack_edge_index = torch.LongTensor(attack_edge_index)

    real_edge_index = np.stack(ori_adj.nonzero())
    real_edge_index = real_edge_index[:, real_edge_index[0, :] < real_edge_index[1, :]]
    real_edge_index = torch.LongTensor(real_edge_index)
    y_preds = []
    for src, dst in torch.hstack((real_edge_index, attack_edge_index)).T:
        feat_src = features[src].toarray()[0]
        feat_dst = features[dst].toarray()[0]
        cosine = np.dot(feat_src, feat_dst)/(np.linalg.norm(feat_src)*np.linalg.norm(feat_dst))
        if np.isnan(cosine): cosine=0
        y_preds.append(cosine)
    y_trues = torch.LongTensor([1] * real_edge_index.shape[1] + [0] * attack_edge_index.shape[1])
    y_preds = torch.tensor(y_preds)
    y_preds = (y_preds+1)/2
    return roc_auc_score(y_trues, y_preds)   

def jaccard_score(ori_adj, attack_adj, features):

    attack_edge_index = np.stack(((attack_adj - ori_adj)>0).nonzero())
    attack_edge_index = attack_edge_index[:, attack_edge_index[0, :] < attack_edge_index[1, :]]
    attack_edge_index = torch.LongTensor(attack_edge_index)

    real_edge_index = np.stack(ori_adj.nonzero())
    real_edge_index = real_edge_index[:, real_edge_index[0, :] < real_edge_index[1, :]]
    real_edge_index = torch.LongTensor(real_edge_index)
    y_preds = []
    for src, dst in torch.hstack((real_edge_index, attack_edge_index)).T:
        feat_src = features[src].toarray()[0]
        feat_dst = features[dst].toarray()[0]
        intersection = np.count_nonzero(feat_src * feat_dst)
        jaccard = intersection / (np.count_nonzero(feat_src) + np.count_nonzero(feat_dst) - intersection + 1e-12)
        y_preds.append(jaccard)
    y_trues = torch.LongTensor([1] * real_edge_index.shape[1] + [0] * attack_edge_index.shape[1])
    y_preds = torch.tensor(y_preds)
    return roc_auc_score(y_trues, y_preds)   

def ks_test_degree(adj, attack_adj, device):
    adj_th = torch.LongTensor(adj.toarray()).to(device)
    attack_adj_th = torch.LongTensor(attack_adj.toarray()).to(device)
    num_nodes = adj_th.shape[0]
    max_degree = num_nodes - 1
    
    unique_degree, unique_degree_count = torch.unique(adj_th.sum(dim=1), return_counts=True)
    dist = torch.zeros(max_degree).to(device)
    dist[unique_degree] = unique_degree_count.to(torch.float)
    acc_dist = torch.cumsum(dist, dim=0)

    unique_attack_degree, unique_attack_degree_count = torch.unique(attack_adj_th.sum(dim=1), return_counts=True)
    attack_dist = torch.zeros(max_degree).to(device)
    attack_dist[unique_attack_degree] = unique_attack_degree_count.to(torch.float)
    attack_acc_dist = torch.cumsum(attack_dist, dim=0)
    return ((acc_dist - attack_acc_dist).abs().max() / num_nodes).item()

def ks_test_lc(adj, attack_adj, device, num_bins=100):
    adj_th = torch.LongTensor(adj.toarray()).to(device)
    attack_adj_th = torch.LongTensor(attack_adj.toarray()).to(device)
    num_nodes = adj_th.shape[0]

    node_deg = adj_th.sum(dim=1).to(torch.float)
    node_denom = node_deg * (node_deg - 1)
    node_numer = torch.zeros_like(node_denom)
    for i in range(num_nodes):
        neigh_n = torch.where(adj_th[i] == 1)[0]
        node_numer[i] = adj_th[neigh_n][:, neigh_n].sum()
    node_lc = torch_safe_division(node_numer, node_denom)
    dist = torch.zeros(num_bins + 1).to(device)
    unique_lc, unique_lc_count = torch.unique((node_lc * num_bins).to(torch.long), return_counts=True)
    dist[unique_lc] = unique_lc_count.to(torch.float)
    acc_dist = torch.cumsum(dist, dim=0)

    attack_node_deg = attack_adj_th.sum(dim=1).to(torch.float)
    attack_node_denom = attack_node_deg * (attack_node_deg - 1)
    attack_node_numer = torch.zeros_like(attack_node_denom)
    for i in range(num_nodes):
        neigh_n = torch.where(attack_adj_th[i] == 1)[0]
        attack_node_numer[i] = attack_adj_th[neigh_n][:, neigh_n].sum()
    attack_node_lc = torch_safe_division(attack_node_numer, attack_node_denom)
    attack_dist = torch.zeros(num_bins + 1).to(device)
    unique_attack_lc, unique_attack_lc_count = torch.unique((attack_node_lc * num_bins).to(torch.long), return_counts=True)
    attack_dist[unique_attack_lc] = unique_attack_lc_count.to(torch.float)
    attack_acc_dist = torch.cumsum(attack_dist, dim=0)
    return ((acc_dist - attack_acc_dist).abs().max() / num_nodes).item()

def llr_test_degree(adj, attack_adj, device):
    d_min = torch.tensor(2.0).to(device)   
    adj_th = torch.LongTensor(adj.toarray()).to(device)
    attack_adj_th = torch.LongTensor(attack_adj.toarray()).to(device)

    node_deg = adj_th.sum(dim=1).to(torch.float)
    attack_node_deg = attack_adj_th.sum(dim=1).to(torch.float)
    concat_node_deg = torch.cat((attack_node_deg, node_deg))

    ll_orig, _, _, _ = node_deg_llr(node_deg, d_min)
    ll_current, _, _, _ = node_deg_llr(attack_node_deg, d_min)
    ll_comb, _, _, _ = node_deg_llr(concat_node_deg, d_min)
    return (-2 * ll_comb + 2 * (ll_orig + ll_current)).item()

def ks_test_homophily(adj, attack_adj, feat, device, num_bins=100):
    adj_th = torch.FloatTensor(adj.toarray()).to(device)
    attack_adj_th = torch.FloatTensor(attack_adj.toarray()).to(device)
    feat_th = torch.FloatTensor(feat.toarray()).to(device)
    num_nodes = adj_th.shape[0]

    node_deg = adj_th.sum(dim=1).to(torch.float)
    node_deg_norm = torch.sqrt(node_deg)
    node_deg_norm_inv = torch_safe_division(torch.ones_like(node_deg_norm), node_deg_norm)
    norm_adj = torch.diag(node_deg_norm_inv) @ adj_th @ torch.diag(node_deg_norm_inv)
    aggr_feat = norm_adj @ feat_th
    node_homo = cosine_sim(feat_th, aggr_feat)
    dist = torch.zeros(num_bins + 1).to(device)
    unique_homo, unique_homo_count = torch.unique((node_homo * num_bins).to(torch.long), return_counts=True)
    dist[unique_homo] = unique_homo_count.to(torch.float)
    acc_dist = torch.cumsum(dist, dim=0)

    attack_node_deg = attack_adj_th.sum(dim=1).to(torch.float)
    attack_node_deg_norm = torch.sqrt(attack_node_deg)
    attack_node_deg_norm_inv = torch_safe_division(torch.ones_like(attack_node_deg_norm), attack_node_deg_norm)
    attack_norm_adj = torch.diag(attack_node_deg_norm_inv) @ attack_adj_th @ torch.diag(attack_node_deg_norm_inv)
    attack_aggr_feat = attack_norm_adj @ feat_th
    attack_node_homo = cosine_sim(feat_th, attack_aggr_feat)
    attack_dist = torch.zeros(num_bins + 1).to(device)
    unique_attacked_homo, unique_attacked_homo_count = torch.unique((attack_node_homo * num_bins).to(torch.long), return_counts=True)
    attack_dist[unique_attacked_homo] = unique_attacked_homo_count.to(torch.float)
    attack_acc_dist = torch.cumsum(attack_dist, dim=0)
    return ((acc_dist - attack_acc_dist).abs().max() / num_nodes).item()

def node_deg_llr(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees

def compute_alpha(n, sum_log_degrees, d_min):
    return  1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))

def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    return n * torch.log(alpha) + n * alpha * torch.log(d_min) - (alpha + 1) * sum_log_degrees

    if len(a.shape) == 1:
        numer = (a * b).sum()
        denom = (torch.norm(a) * torch.norm(b))
    else:
        numer = (a * b).sum(dim=1)
        denom = (torch.norm(a, dim=1) * torch.norm(b, dim=1))
    return torch_safe_division(numer, denom)  