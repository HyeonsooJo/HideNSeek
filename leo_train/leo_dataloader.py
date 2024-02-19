from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import numpy as np
import torch
import os
import pdb

def load_lp_dataset(train_data, valid_data, adj, attack_adj, features, batch_size=256):
    # mp_adj
    mp_adj = torch.zeros(adj.shape)
    mp_adj[train_data.edge_index[0], train_data.edge_index[1]] = 1
    
    # norm_adj
    norm_adj = mp_adj + torch.eye(mp_adj.shape[0])
    rowsum = norm_adj.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    norm_adj = r_mat_inv @ norm_adj @ r_mat_inv
    
    # features
    features = torch.Tensor(features.toarray())
    
    # norm_features
    if adj.shape[0] == 7624: # lastfmasia
        norm_features = features
    else:
        norm_features = train_data.x

    # train_loader
    train_dataset = lp_dataset(train_data.pos_edge_label_index, train_data.pos_edge_label,
                                train_data.neg_edge_label_index, train_data.neg_edge_label)
    train_loader = lp_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Valid_loader
    valid_dataset = lp_dataset(valid_data.pos_edge_label_index, valid_data.pos_edge_label,
                                valid_data.neg_edge_label_index, valid_data.neg_edge_label)
    valid_loader = lp_dataloader(valid_dataset, batch_size=batch_size)
    
    
    # test_loader
    if attack_adj == None:
        return  (mp_adj, norm_adj, features, norm_features), train_loader, valid_loader
    else:
        attack_add_edge_index = np.stack(((attack_adj - adj) > 0).nonzero())
        attack_add_edge_index = attack_add_edge_index[:, attack_add_edge_index[0, :] < attack_add_edge_index[1, :]]
        attack_add_edge_index = torch.LongTensor(attack_add_edge_index)

        real_edge_index = np.stack(adj.nonzero())
        real_edge_index = real_edge_index[:, real_edge_index[0, :] < real_edge_index[1, :]]
        real_edge_index = torch.LongTensor(real_edge_index)
        
        pos_edge_index = real_edge_index

        neg_edge_index = attack_add_edge_index
        pos_label = torch.Tensor([1] * pos_edge_index.shape[1])
        neg_label = torch.Tensor([0] * neg_edge_index.shape[1])
        
        test_dataset = lp_dataset(pos_edge_index, pos_label, neg_edge_index, neg_label)
        test_loader = lp_dataloader(test_dataset, batch_size=batch_size)
        return (mp_adj, norm_adj, features, norm_features), train_loader, valid_loader, test_loader

def load_lp_dataset_sparse(train_data, valid_data, adj, attack_adj, features, batch_size):
    # mp_adj
    mp_adj = torch.zeros(adj.shape)
    mp_adj[train_data.edge_index[0], train_data.edge_index[1]] = 1
    
    # norm_adj
    norm_adj = mp_adj + torch.eye(mp_adj.shape[0])
    rowsum = norm_adj.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    norm_adj = r_mat_inv @ norm_adj @ r_mat_inv
    
    # features
    features = torch.Tensor(features.toarray())
    
    # norm_features
    norm_features = train_data.x
    
    # train_loader
    train_dataset = lp_dataset(train_data.pos_edge_label_index, train_data.pos_edge_label,
                                train_data.neg_edge_label_index, train_data.neg_edge_label)
    train_loader = lp_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Valid_loader
    valid_dataset = lp_dataset(valid_data.pos_edge_label_index, valid_data.pos_edge_label,
                                valid_data.neg_edge_label_index, valid_data.neg_edge_label)
    valid_loader = lp_dataloader(valid_dataset, batch_size=batch_size)
    
    
    # test_loader
    if attack_adj == None:
        return  (mp_adj, norm_adj, features, norm_features), train_loader, valid_loader
    else:
        attack_add_edge_index = np.stack(((attack_adj - adj) > 0).nonzero())
        attack_add_edge_index = attack_add_edge_index[:, attack_add_edge_index[0, :] < attack_add_edge_index[1, :]]
        attack_add_edge_index = torch.LongTensor(attack_add_edge_index)

        real_edge_index = np.stack(adj.nonzero())
        real_edge_index = real_edge_index[:, real_edge_index[0, :] < real_edge_index[1, :]]
        real_edge_index = torch.LongTensor(real_edge_index)
        
#         attack_del_edge_index = np.stack(((attack_adj - adj) < 0).nonzero())
#         num_del = np.stack(((attack_adj - adj) < 0).nonzero()).shape[-1]

#         if num_del > 0:
#             attack_del_edge_index = attack_del_edge_index[:, attack_del_edge_index[0, :] < attack_del_edge_index[1, :]]
#             attack_del_edge_index = torch.LongTensor(attack_del_edge_index)
#             pos_edge_index = torch.cat([real_edge_index, attack_del_edge_index], dim=1)
#         else:
        pos_edge_index = real_edge_index

        neg_edge_index = attack_add_edge_index
        pos_label = torch.LongTensor([1] * pos_edge_index.shape[1])
        neg_label = torch.LongTensor([0] * neg_edge_index.shape[1])
        
        test_dataset = lp_dataset(pos_edge_index, pos_label, neg_edge_index, neg_label)
        test_loader = lp_dataloader(test_dataset, batch_size=batch_size)
        return (mp_adj, norm_adj, features, norm_features), train_loader, valid_loader, test_loader
     

class lp_dataset(Dataset):
    def __init__(self, pos_edge_index, pos_label, neg_edge_index, neg_label):
        self.edge_list = torch.cat([pos_edge_index, neg_edge_index], dim=1).T
        self.y = torch.cat([pos_label, neg_label])
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.edge_list[idx], self.y[idx]
    
class lp_dataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(lp_dataloader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
def _collate_fn(batch):
    edge_index_batch = torch.vstack([edge_index for edge_index, _ in batch])
    y_batch = torch.vstack([y for _, y in batch]) 
    return edge_index_batch, y_batch