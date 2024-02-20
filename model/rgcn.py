from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.attack_utils  
import torch.optim as optim
from copy import deepcopy

# TODO sparse implementation

class GGCL_F(Module):
    """Graph Gaussian Convolution Layer (GGCL) when the input is feature"""

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GGCL_F, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, features, adj_norm1, adj_norm2, gamma=1):
        features = F.dropout(features, self.dropout, training=self.training)
        self.miu = F.elu(torch.mm(features, self.weight_miu))
        self.sigma = F.relu(torch.mm(features, self.weight_sigma))
        # torch.mm(previous_sigma, self.weight_sigma)
        Att = torch.exp(-gamma * self.sigma)
        miu_out = adj_norm1 @ (self.miu * Att)
        sigma_out = adj_norm2 @ (self.sigma * Att * Att)
        return miu_out, sigma_out

class GGCL_D(Module):

    """Graph Gaussian Convolution Layer (GGCL) when the input is distribution"""
    def __init__(self, in_features, out_features, dropout):
        super(GGCL_D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        # self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, miu, sigma, adj_norm1, adj_norm2, gamma=1):
        miu = F.dropout(miu, self.dropout, training=self.training)
        sigma = F.dropout(sigma, self.dropout, training=self.training)
        miu = F.elu(miu @ self.weight_miu)
        sigma = F.relu(sigma @ self.weight_sigma)

        Att = torch.exp(-gamma * sigma)
        mean_out = adj_norm1 @ (miu * Att)
        sigma_out = adj_norm2 @ (sigma * Att * Att)
        return mean_out, sigma_out


class RGCN(Module):
    """Robust Graph Convolutional Networks Against Adversarial Attacks. KDD 2019.

    Parameters
    ----------
    nnodes : int
        number of nodes in the input grpah
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    gamma : float
        hyper-parameter for RGCN. See more details in the paper.
    beta1 : float
        hyper-parameter for RGCN. See more details in the paper.
    beta2 : float
        hyper-parameter for RGCN. See more details in the paper.
    lr : float
        learning rate for GCN
    dropout : float
        dropout rate for GCN
    device: str
        'cpu' or 'cuda'.

    """

    def __init__(self, nnodes, nfeat, nhid, nclass, gamma=1.0, beta1=5e-4, beta2=5e-4, lr=0.01, dropout=0.6, weight_decay=0.005, device='cpu'):
        super(RGCN, self).__init__()

        self.device = device
        # adj_norm = normalize(adj)
        # first turn original features to distribution
        self.lr = lr
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.nclass = nclass
        self.nhid = nhid // 2
        self.gc1 = GGCL_F(nfeat, nhid, dropout=dropout)
        self.gc2 = GGCL_D(nhid, nclass, dropout=dropout)

        self.dropout = dropout
        self.weight_decay = weight_decay
        # self.gaussian = MultivariateNormal(torch.zeros(self.nclass), torch.eye(self.nclass))
        self.gaussian = MultivariateNormal(torch.zeros(nnodes, nclass).to(device),
                                torch.diag_embed(torch.ones(nnodes, nclass)).to(device))
        
        self.adj_norm1, self.adj_norm2 = None, None
        self.features, self.labels = None, None

    def forward(self, eval_flag=False):
        features = self.features
        miu, sigma = self.gc1(features, self.adj_norm1, self.adj_norm2, self.gamma)
        miu, sigma = self.gc2(miu, sigma, self.adj_norm1, self.adj_norm2, self.gamma)
        if eval_flag:
            embeddings = miu
        else:
            embeddings = miu + self.gaussian.sample().to(self.device) * torch.sqrt(sigma + 1e-8)

        output = miu + self.gaussian.sample().to(self.device) * torch.sqrt(sigma + 1e-8)
        return F.log_softmax(output, dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=False, patience=500, **kwargs):
        """Train RGCN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        verbose : bool
            whether to show verbose logs

        Examples
        --------
        We can first load dataset and then train RGCN.

        >>> from deeprobust.graph.data import PrePtbDataset, Dataset
        >>> from deeprobust.graph.defense import RGCN
        >>> # load clean graph data
        >>> data = Dataset(root='/tmp/', name='cora', seed=15)
        >>> adj, features, labels = data.adj, data.features, data.labels
        >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        >>> # load perturbed graph data
        >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
        >>> perturbed_adj = perturbed_data.adj
        >>> # train defense model
        >>> model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1],
                         nclass=labels.max()+1, nhid=32, device='cpu')
        >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val,
                      train_iters=200, verbose=True)
        >>> model.test(idx_test)

        """
        adj, norm_features, labels = utils.preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=True, sparse=False, device=self.device)

        self.features, self.labels = norm_features, labels
        self.adj_norm1 = self._normalize_adj(adj, power=-1/2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)
        
        self._initialize()
        self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def set_data(self, features, adj, labels):
        adj, norm_features, labels = utils.preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=True, sparse=False, device=self.device)
        
        self.features, self.labels = norm_features, labels
        self.adj_norm1 = self._normalize_adj(adj, power=-1/2)
        self.adj_norm2 = self._normalize_adj(adj, power=-1)

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose: print('=== training rgcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(eval_flag=True)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate the peformance on test set
        """
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of RGCN
        """

        self.eval()
        return self.forward(eval_flag=True)

    def _loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        miu1 = self.gc1.miu
        sigma1 = self.gc1.sigma
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-8 + sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + \
                torch.norm(self.gc1.weight_sigma, 2).pow(2)
        return loss  + self.beta1 * kl_loss + self.beta2 * norm2

    def _initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def _normalize_adj(self, adj, power=-1/2):

        """Row-normalize sparse matrix"""
        A = adj + torch.eye(len(adj)).to(self.device)
        D_power = (A.sum(1)).pow(power)
        D_power[torch.isinf(D_power)] = 0.
        D_power = torch.diag(D_power)
        return D_power @ A @ D_power
