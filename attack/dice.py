from .baseattack import BaseAttack

import scipy.sparse as sp
import numpy as np
import random
import torch


class DICE(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(DICE, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        assert not self.attack_features, 'DICE does NOT support attacking features'

    def attack(self, ori_features, ori_adj, labels, n_perturbations, **kwargs):
        remove_or_insert = np.random.choice(2, n_perturbations)
        n_remove = sum(remove_or_insert)
    
        indices = sp.triu(ori_adj.tolil()).nonzero()
        possible_indices = [x for x in zip(indices[0], indices[1]) if labels[x[0]] == labels[x[1]]]
        remove_indices = np.random.permutation(possible_indices)[: min(n_remove, ori_adj.sum())]

        n_insert = n_perturbations - n_remove
    
        # sample edges to add
        modified_adj = ori_adj.copy().tolil()        
        candidate_edges = []
        
        while len(candidate_edges) < n_insert:           
            # sample random pairs
            cand_edges = np.array([np.random.choice(ori_adj.shape[0], n_insert),
                                        np.random.choice(ori_adj.shape[0], n_insert)]).T
            # filter out existing edges, and pairs with the different labels
            candidate_edges += list(set([(u, v) for u, v in cand_edges if labels[u] != labels[v]
                                        and modified_adj[u, v] == 0 and modified_adj[v, u] == 0 and u<v]))

            
        candidate_edges = candidate_edges[0:n_insert]
        candidate_edges += list([(u,v) for u,v in remove_indices])
        candidate_edges = np.array([[i, j] for i, j in candidate_edges])

        cur_adj_th = torch.Tensor(ori_adj.toarray()).to(self.device)
        cur_adj_th[candidate_edges[:, 0], candidate_edges[:, 1]] = 1 - cur_adj_th[candidate_edges[:, 0], candidate_edges[:, 1]]
        cur_adj_th[candidate_edges[:, 1], candidate_edges[:, 0]] = 1 - cur_adj_th[candidate_edges[:, 1], candidate_edges[:, 0]]

        self.check_adj(cur_adj_th)
        self.modified_adj = sp.csr_matrix(cur_adj_th.detach().cpu().numpy())
        
            
    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]