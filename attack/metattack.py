from .baseattack import BaseAttack
from utils import * 

import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
import torch
import math
import tqdm
import warnings
warnings.filterwarnings("ignore", message="UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch.")


class BaseMeta(BaseAttack):
    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, undirected=True, device='cpu'):

        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = nn.Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)

        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = adj_changes_square.to('cpu') + ori_adj.to('cpu')
        return modified_adj
    
    def filter_potential_singletons(self, modified_adj):
        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    def get_adj_score(self, adj_grad, modified_adj):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        adj_meta_grad -= adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask
        return adj_meta_grad
     

class Metattack(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.
    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True,
                 device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):

        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = nn.Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = nn.Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = nn.Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = nn.Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, norm_adj, idx_train, idx_unlabeled, labels):
        self._initialize()
        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = norm_adj @ torch.spmm(hidden, w) + b
                else:
                    hidden = norm_adj @ hidden @ w + b
                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, norm_adj, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = norm_adj @ torch.spmm(hidden, w) + b
            else:
                hidden = norm_adj @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        adj_grad = None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        
        return adj_grad
                
    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled,
               n_perturbations, **kwargs):


        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = to_tensor(ori_adj, ori_features, labels, device=self.device)

        labels_self_training = self.self_training_label(labels, idx_train)

        modified_adj = ori_adj.clone()
        modified_features = ori_features.clone()
        candidate_edges = []
        for i in tqdm.tqdm(range(n_perturbations)):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj).to(self.device)     # get_modified_adj       
            norm_modified_adj = normalize_adj_tensor(modified_adj)
            
            self.inner_train(modified_features, norm_modified_adj, idx_train, idx_unlabeled, labels)
            
            adj_grad = self.get_meta_grad(modified_features, norm_modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            adj_meta_score = self.get_adj_score(adj_grad.detach(), modified_adj.detach())
            adj_meta_argmax = torch.argmax(adj_meta_score)

            row_idx, col_idx = unravel_index(adj_meta_argmax, ori_adj.shape)
            
            while (row_idx.to('cpu').item(), col_idx.to('cpu').item()) in candidate_edges or \
                (col_idx.to('cpu').item(), row_idx.to('cpu').item()) in candidate_edges:
                adj_meta_score[row_idx, col_idx] = 0
                adj_meta_score[col_idx, row_idx] = 0
                if (adj_meta_score==0).sum() == adj_meta_score.shape[0]*adj_meta_score.shape[1]: break
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = unravel_index(adj_meta_argmax, ori_adj.shape)
            if (adj_meta_score==0).sum() == adj_meta_score.shape[0]*adj_meta_score.shape[1]: continue
            candidate_edges.append((row_idx.to('cpu').numpy(), col_idx.to('cpu').numpy()))
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            if self.undirected:
                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                
        ori_adj = ori_adj.to_dense()
        cur_adj_th = ori_adj.clone().detach()

        candidate_edges = np.array([[i, j] for i, j in candidate_edges])

        cur_adj_th = torch.Tensor(ori_adj.detach().cpu().numpy()).to(self.device)
        cur_adj_th[candidate_edges[:, 0], candidate_edges[:, 1]] = 1 - cur_adj_th[candidate_edges[:, 0], candidate_edges[:, 1]]
        cur_adj_th[candidate_edges[:, 1], candidate_edges[:, 0]] = 1 - cur_adj_th[candidate_edges[:, 1], candidate_edges[:, 0]]

        self.check_adj(cur_adj_th)
        self.modified_adj = sp.csr_matrix(cur_adj_th.detach().cpu().numpy()) 

