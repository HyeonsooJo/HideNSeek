from torch_geometric.datasets import Planetoid, LastFMAsia, CitationFull
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

import scipy.sparse as sp
import numpy as np
import random
import torch
import json
import os

from .metric_utils import jaccard_fn, svd_fn, get_all_jaccard, cosine_fn
from .attack_utils import normalize_feature


def load_dataset(root, graph_name):
    if graph_name == 'cora' or graph_name == 'citeseer':
        dataset = Planetoid(root=os.path.join(root, graph_name), 
                            name=graph_name, 
                            split='public')
    elif graph_name == 'lastfmasia':
        dataset = LastFMAsia(root=os.path.join(root, graph_name))
    elif graph_name == 'cora_ml':
        dataset = CitationFull(root=os.path.join(root, graph_name), 
                            name=graph_name)
    elif graph_name == 'chameleon' or graph_name == 'squirrel':
        data = np.load(os.path.join(root, graph_name, graph_name, graph_name + '_filtered.npz'))

        
    if graph_name == 'chameleon' or graph_name == 'squirrel':
        edges = data['edges']
        node_features = data['node_features']
        labels = data['node_labels']
        train_mask = data['train_masks'][0]
        val_mask = data['val_masks'][0]
        test_mask  = data['test_masks'][0]
        num_nodes = node_features.shape[0]
        adj = np.zeros([num_nodes, num_nodes])
        adj[edges[:, 0], edges[:, 1]] = 1
        adj[edges[:, 1], edges[:, 0]] = 1
        features = node_features
        adj = sp.csr_matrix(adj)
        features = sp.csr_matrix(features)

    else:    
        data = dataset[0]

        num_nodes = data.x.shape[0]
        features = sp.csr_matrix(data.x)

        origin_row, origin_col = data.edge_index[0].numpy(), data.edge_index[1].numpy()
        origin_adj = sp.csr_matrix((np.ones_like(origin_row), (origin_row, origin_col)), shape=(num_nodes, num_nodes))
        origin_adj_np = origin_adj.toarray()
        np.fill_diagonal(origin_adj_np, 0)
        origin_adj_np = origin_adj_np + origin_adj_np.T
        row, col = np.nonzero(origin_adj_np)
        value = np.ones_like(row)
        adj = sp.csr_matrix((value, (row, col)), shape=(num_nodes, num_nodes))
        labels = data.y.numpy().astype(np.int8)
        
        if graph_name == 'cora' or graph_name == 'citeseer':
            train_mask = dataset[0].train_mask
            val_mask = dataset[0].val_mask
            test_mask = dataset[0].test_mask


        elif graph_name == 'lastfmasia':
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            for label in np.unique(labels):
                mask_label = np.where(labels == label)[0]
                num_train = int(0.05 * len(mask_label))
                num_val = num_train
                train_mask[mask_label[:num_train]] = True
                val_mask[mask_label[num_train: num_train + num_val]] = True
                test_mask[mask_label[num_train + num_val:]] = True

        if graph_name == 'lastfmasia':
            np.random.seed(0)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)            

            for label in np.unique(labels):
                mask_label = np.where(labels == label)[0]
                num_train = int(0.05 * len(mask_label))
                num_val = num_train
                train_mask[mask_label[:num_train]] = True
                val_mask[mask_label[num_train: num_train + num_val]] = True
                test_mask[mask_label[num_train + num_val:]] = True
        elif graph_name == 'cora_ml':
            np.random.seed(0) 
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            for label in np.unique(labels):
                mask_label = np.where(labels == label)[0]
                train_mask[mask_label[:20]] = True
            
            val_cand_idx = np.where(train_mask == False)[0]
            val_idx = np.random.choice(val_cand_idx, 500, replace=False)
            val_mask[val_idx] = True
            
            test_cand_idx = np.where((train_mask + val_mask) == False)[0]
            test_idx = np.random.choice(test_cand_idx, 1000, replace=False)
            test_mask[test_idx] = True

    idx_train = np.where(train_mask == True)[0]
    idx_val = np.where(val_mask == True)[0]
    idx_test = np.where(test_mask == True)[0]
    idx_unlabeled = np.union1d(idx_val, idx_test)
    
    if (adj.sum(axis=0)==0).sum() > 0:
        lcc_idx = np.where(np.array(adj.sum(axis=0)).squeeze()>0)[0]
        adj = adj[:,lcc_idx][lcc_idx]
        features = features[lcc_idx]
        labels = labels[lcc_idx]

        idx_train = np.where(train_mask[lcc_idx] == True)[0]
        idx_val = np.where(val_mask[lcc_idx] == True)[0]
        idx_test = np.where(test_mask[lcc_idx] == True)[0]
        idx_unlabeled = np.union1d(idx_val, idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, idx_unlabeled


def load_split_data(root, graph_name, attack_info, seed, adj, features, filtered=None):
    ###############        SEED        ###############
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)
    ##################################################
    if attack_info == None:
        data_path = os.path.join(root, graph_name, 'clean', 'split_data')
        train_path = os.path.join(data_path, f'train_data_seed{seed}.pt')
        valid_path =os.path.join(data_path, f'valid_data_seed{seed}.pt')
        if not (os.path.isfile(train_path) and os.path.isfile(valid_path)):
            print('No split data for clean graph')
            train_data, valid_data = split_data(adj, features)
            print('Saving...')
            os.makedirs(os.path.dirname(train_path), exist_ok=True)            
            torch.save(train_data, train_path)
            torch.save(valid_data, valid_path)
        else:
            train_data, valid_data = torch.load(train_path), torch.load(valid_path)
    else:
        data_path = os.path.join(root, graph_name, 'attack', 'split_data')
        train_path = os.path.join(data_path, f'train_data_{attack_info}_seed{seed}.pt')
        valid_path =os.path.join(data_path, f'valid_data_{attack_info}_seed{seed}.pt')
        # train_data, valid_data = split_data(adj, features)
        if not (os.path.isfile(train_path) and os.path.isfile(valid_path)):
            print('No split data for attacked graph')
            train_data, valid_data = split_data(adj, features)
            print('Saving...')
            os.makedirs(os.path.dirname(train_path), exist_ok=True)            
            torch.save(train_data, train_path)
            torch.save(valid_data, valid_path)
        else:
            train_data, valid_data = torch.load(train_path), torch.load(valid_path)

    train_data = Data(x=train_data.x, edge_index=train_data.edge_index, 
            pos_edge_label=train_data.pos_edge_label,
            pos_edge_label_index=train_data.pos_edge_label_index,
            neg_edge_label=train_data.neg_edge_label,
            neg_edge_label_index=train_data.neg_edge_label_index)
    valid_data = Data(x=valid_data.x, edge_index=valid_data.edge_index, 
            pos_edge_label=valid_data.pos_edge_label,
            pos_edge_label_index=valid_data.pos_edge_label_index, 
            neg_edge_label=valid_data.neg_edge_label,
            neg_edge_label_index=valid_data.neg_edge_label_index)
    return train_data, valid_data

def split_data(attack_adj, features, num_val=0.2):
    # attack_adj는 csr matrix
    if torch.is_tensor(attack_adj):
        if attack_adj.is_sparse: attack_adj = attack_adj.to_dense()
        attack_adj = attack_adj.numpy()
    if type(attack_adj) != sp.csr.csr_matrix:
        attack_adj = sp.csr_matrix(attack_adj)
    edge_index_th = torch.LongTensor(np.stack(attack_adj.nonzero()))
    norm_features = normalize_feature(features)
    norm_features = torch.FloatTensor(norm_features.toarray())

    data = Data(x=norm_features, edge_index=edge_index_th)
    transform = RandomLinkSplit(num_val=num_val, num_test=0.0,
                                is_undirected=True, split_labels=True, neg_sampling_ratio=1) # 
    train_data, val_data, _ = transform(data)
    return train_data, val_data
