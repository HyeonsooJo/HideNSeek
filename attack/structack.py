from scipy.optimize import linear_sum_assignment
from .baseattack import BaseAttack

import scipy.sparse as sp
import networkx as nx
import numpy as np
import pickle
import torch
import tqdm
import os
import pdb


class Structack(BaseAttack):
    def __init__(self, model=None, nnodes=None, attack_structure=True, attack_features=False, device='cpu'):
        super(Structack, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)
        self.modified_adj = None
        self.node_selection = self.get_nodes_with_lowest_betweenness_centrality
        
    def attack(self, ori_features, ori_adj, n_perturbations, dataset, **kwargs):
        n_selected_nodes = n_perturbations * 2
        graph = nx.from_scipy_sparse_matrix(ori_adj)
        assert ori_adj.shape[0] == graph.number_of_nodes()
        assert ori_adj.sum() / 2 == graph.number_of_edges()
        
        # selection
        nodes = self.node_selection(graph, n_selected_nodes, dataset)
        # connection
        edges = self.node_connection(graph, nodes, min(n_perturbations, ori_adj.shape[0]), dataset)

        cur_adj_th = torch.Tensor(ori_adj.toarray()).to(self.device)
        candidate_edges = np.array([[i, j] for i, j in edges])

        cur_adj_th[candidate_edges[:, 0], candidate_edges[:, 1]] = 1 
        cur_adj_th[candidate_edges[:, 1], candidate_edges[:, 0]] = 1

        self.check_adj(cur_adj_th)
        self.modified_adj = sp.csr_matrix(cur_adj_th.detach().cpu().numpy())
 
    def get_nodes_with_lowest_betweenness_centrality(self, graph, n_selected_nodes, dataset):
        precomputed_path = os.path.join(os.getcwd(), dataset + "_betweenness_centralities.pkl")
        if os.path.exists(precomputed_path):
            print("Loading precomputed_betweenness_centralities...")
            with open(precomputed_path,'r') as ff:
                betweenness_centralities = pickle.load(open(precomputed_path, 'rb'))
        else:
            betweenness_centralities = nx.betweenness_centrality(graph)
            pickle.dump(betweenness_centralities, open(precomputed_path, "wb" ) )

        nodes = sorted(betweenness_centralities.items(), key=lambda x: x[1])
        if len(nodes) < n_selected_nodes:  # repeat the list until it's longer than n
            nodes = nodes * int(n_selected_nodes / len(nodes) + 1)
        nodes = nodes[:n_selected_nodes]
        return [x[0] for x in nodes]

    def node_connection(self, graph, nodes, n_perturbations, dataset, threshold=0.000001, nsteps=10000):
        rows = nodes[:n_perturbations]
        cols = nodes[-n_perturbations:]
        # cols = nodes[n_perturbations:]
        precomputed_path = os.path.join(os.getcwd(), dataset + "_katz.pkl")
        if dataset is not None and os.path.exists(precomputed_path):
            print("Loading precomputed_katz...")
            with open(precomputed_path,'rb') as ff:
                sigma = pickle.load(ff)
        else:
            D = nx.linalg.laplacianmatrix.laplacian_matrix(graph) + nx.to_scipy_sparse_matrix(graph)
            D_inv = sp.linalg.inv(D)
            D_invA = D_inv * nx.to_scipy_sparse_matrix(graph) 
            l,v = sp.linalg.eigs(D_invA, k=1, which="LR")
            lmax = l[0].real
            alpha = (1/lmax) * 0.9
            sigma = sp.csr_matrix(D_invA.shape, dtype=np.float)
            # print('Calculate sigma matrix')
            for i in range(nsteps):
                sigma_new = alpha *D_invA*sigma + sp.identity(graph.number_of_nodes(), dtype=np.float, format='csr')
                diff = abs(sp.linalg.norm(sigma, 1) - sp.linalg.norm(sigma_new, 1))
                sigma = sigma_new
                # print(diff)
                if diff < threshold: 
                    break
                # print('Number of steps taken: ' + str(i))
            sigma = sigma.toarray().astype('float')
            with open(precomputed_path,'wb') as ff:
                pickle.dump(sigma, ff)
        
        mtx = np.array([[sigma[u][v] for v in cols] for u in rows])
        u, v = linear_sum_assignment(+mtx)
        edges = [(rows[i], cols[j]) for i, j in zip(u, v)] 
        return edges
    