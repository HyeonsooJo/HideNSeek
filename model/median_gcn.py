from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim
from copy import deepcopy
from deeprobust.graph import utils
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv import MessagePassing

# This works for higher version of torch_gometric, e.g., 2.0.
# from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear


from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import remove_self_loops, add_self_loops


class MedianConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', None)
        super(MedianConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        # This works for higher version of torch_gometric, e.g., 2.0.
        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=x.size(self.node_dim))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def aggregate(self, x_j, index):
        """median aggregation"""
        # important! `to_dense_batch` requires the `index` is sorted
        ix = torch.argsort(index)
        index = index[ix]
        x_j = x_j[ix]

        dense_x, mask = to_dense_batch(x_j, index)
        out = x_j.new_zeros(dense_x.size(0), dense_x.size(-1))
        deg = mask.sum(dim=1)
        for i in deg.unique():
            deg_mask = deg == i
            out[deg_mask] = dense_x[deg_mask, :i].median(dim=1).values
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MedianGCN(torch.nn.Module):
    """Graph Convolutional Networks with Median aggregation (MedianGCN) 
    based on pytorch geometric. 

    `Understanding Structural Vulnerability in Graph Convolutional Networks 
    <https://arxiv.org/abs/2108.06280>`

    MedianGCN uses median aggregation function instead of 
    `weighted mean` adopted in GCN, which improves the robustness 
    of the model against adversarial structural attack.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units        
    nclass : int
        size of output dimension
    lr : float
        learning rate for MedianGCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for MedianGCN.
    with_bias: bool
        whether to include bias term in MedianGCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
        We can first load dataset and then train MedianGCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import MedianGCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> MedianGCN = MedianGCN(nfeat=features.shape[1],
                          nhid=16, nclass=labels.max().item() + 1, 
                          device='cuda')
    >>> MedianGCN = MedianGCN.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> MedianGCN.fit(pyg_data, verbose=True) # train with earlystopping
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_bias=True, device=None):

        super(MedianGCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = MedianConv(nfeat, nhid, bias=with_bias)
        self.conv2 = MedianConv(nhid, nclass, bias=with_bias)

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.with_bias = with_bias
        self.output = None
        self.edge_index = None
        self.features = None
        self.labels = None


    def forward(self):
        x = F.dropout(self.features, self.dropout, training=self.training)
        x = self.conv1(x, self.edge_index).relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, self.edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of MedianGCN.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=False, patience=500, **kwargs):
        """Train the MedianGCN model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        # self.device = self.conv1.weight.device
        adj, norm_features, labels = utils.preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=True, sparse=False, device=self.device)
        self.edge_index = adj.nonzero().T
        self.features, self.labels = norm_features, labels
        self.initialize()


        # By default, it is trained with early stopping on validation
        self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def set_data(self, features, adj, labels):
        adj, norm_features, labels = utils.preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=True, sparse=False, device=self.device)
        self.edge_index = adj.nonzero().T
        self.features, self.labels = norm_features, labels
        
    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training MedianGCN model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward()

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward()
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

    @torch.no_grad()
    def test(self, idx_test):
        """Evaluate MedianGCN performance on test set.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object        
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
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
        return self.forward()
