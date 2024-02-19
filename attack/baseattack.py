import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn.modules.module import Module

import utils


class BaseAttack(Module):
    def __init__(self, model, nnodes, attack_structure=True, attack_features=False, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

    def attack(self, ori_adj, n_perturbations, **kwargs):
        pass

    def check_adj(self, adj):
        adj_np = adj.detach().cpu().numpy()
        assert np.abs(adj_np - adj_np.T).sum() == 0, "Input graph is not symmetric"


    def check_adj_tensor(self, adj):
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"


    def save_adj(self, root=r'/tmp/', name='mod_adj'):
        assert self.modified_adj is not None, \
                'modified_adj is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_adj = self.modified_adj

        if type(modified_adj) is torch.Tensor:
            sparse_adj = utils.to_scipy(modified_adj)
            sp.save_npz(osp.join(root, name), sparse_adj)
        else:
            sp.save_npz(osp.join(root, name), modified_adj)

    def save_features(self, root=r'/tmp/', name='mod_features'):
        assert self.modified_features is not None, \
                'modified_features is None! Please perturb the graph first.'
        name = name + '.npz'
        modified_features = self.modified_features

        if type(modified_features) is torch.Tensor:
            sparse_features = utils.to_scipy(modified_features)
            sp.save_npz(osp.join(root, name), sparse_features)
        else:
            sp.save_npz(osp.join(root, name), modified_features)