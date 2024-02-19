from .baseattack import BaseAttack

import scipy.sparse as sp
import numpy as np
import torch


class Random(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Random, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'RND does NOT support attacking features'

    def attack(self, ori_features, ori_adj, n_perturbations, **kwargs):
        if self.attack_structure:
            # self.modified_adj: sp.csr_matrix
            self.modified_adj = self.perturb_adj(ori_adj, n_perturbations)
            
        print((self.modified_adj - ori_adj).sum()//2)
        
    def perturb_adj(self, adj, n_perturbations):
        cur_adj_th = torch.Tensor(adj.toarray()).to(self.device)

        nonzero = set(zip(*(adj).nonzero()))
        candidate_edges = self.random_sample_edges(adj, n=n_perturbations, exclude=nonzero)
        candidate_edges = np.array([[i, j] for i, j in candidate_edges])
        cur_adj_th[candidate_edges[:, 0], candidate_edges[:, 1]] = 1
        cur_adj_th[candidate_edges[:, 1], candidate_edges[:, 0]] = 1
        self.check_adj(cur_adj_th)
        modified_adj = sp.csr_matrix(cur_adj_th.detach().cpu().numpy())
        return modified_adj

    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))
