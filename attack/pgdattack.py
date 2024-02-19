from .baseattack import BaseAttack

import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
import torch
import tqdm

import utils


class PGDAttack(BaseAttack):
    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False,  device='cpu'):

        super(PGDAttack, self).__init__(model, nnodes, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = nn.Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_train, n_perturbations, epochs=200, **kwargs):
        victim_model = self.surrogate
        norm_features = utils.normalize_feature(ori_features)
        ori_adj, norm_features, labels = utils.to_tensor(ori_adj, norm_features, labels, device=self.device)
        
        victim_model.eval()
        for t in tqdm.tqdm(range(epochs), leave=False):
            modified_adj = self.get_modified_adj(ori_adj)
            norm_modified_adj = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(norm_features, norm_modified_adj)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection(n_perturbations)

        self.random_sample_without_budget(ori_adj, norm_features, labels, idx_train, n_perturbations)
        modified_adj = self.get_modified_adj(ori_adj)
        modified_adj = modified_adj.clone().detach().to('cpu').numpy()
        self.modified_adj = sp.csr_matrix(modified_adj)
        

    def random_sample_without_budget(self, ori_adj, norm_features, labels, idx_train, n_perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)
                if sampled.sum() > n_perturbations:
                    continue
                
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                norm_modified_adj = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(norm_features, norm_modified_adj)
                loss = self._loss(output[idx_train], labels[idx_train])
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self, ori_adj):
        ori_adj = ori_adj.to('cpu')
        if self.complementary is None:
            # self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
            self.complementary = (torch.ones(ori_adj.shape) - torch.eye(self.nnodes) - ori_adj) - ori_adj # HJ fixed

        m = torch.zeros((self.nnodes, self.nnodes))
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes.to('cpu')
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj.to(self.device)

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu