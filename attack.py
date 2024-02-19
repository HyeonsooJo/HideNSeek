from genericpath import isfile
from copy import deepcopy

import numpy as np
import argparse
import random
import torch
import os

from attack import *
from lp_train import *
from model import *
from utils import *

import warnings
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='Comparing a sparse matrix with 0 using == is inefficient, try using != instead.')
    ###############     Arguments     ###############
    parser = argparse.ArgumentParser()   
    parser.add_argument('--gpu', type=int, default=0, help='cpu if -1')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--root', type=str, default=os.path.join(os.getcwd(), 'data'), help='save data path')

    # Attack settings
    parser.add_argument('--graph_name', type=str, default='cora', help='graph name: cora, citeeer,...')
    parser.add_argument('--attacker', type=str, default='random', help='attacker model name: random, dice, pgd, ...')
    parser.add_argument('--attack_rate', type=int, default=10, help='percentage of perturbing edges')
        
    args = parser.parse_args()
    args.device = 'cpu' if args.gpu==-1 else 'cuda'
    
    nc_config = json.load(open(os.path.join('config', 'gcn_nc.json')))
    adj, features, labels, idx_train, idx_val, idx_test, idx_unlabeled = load_dataset(root=args.root, graph_name=args.graph_name)

    args.nfeat = features.shape[1]
    args.nclass = labels.max().item() + 1
    print(f'Loading {args.graph_name} with {len(labels)} nodes and {adj.sum()/2} edges.')    


    ###############        SEED        ###############
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(args.seed)    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    ###############     Save data     ###############
    attack_info = f'attacked_adj_{args.attacker}_{args.attack_rate}'
    path_attacked_adj =  f'{args.root}/{args.graph_name}/attacked_graph/{attack_info}_seed{args.seed}.pt'
    os.makedirs(f'{args.root}/{args.graph_name}/attacked_graph', exist_ok=True)

    ############### Node classification ###############
    victim_model = GCN(nfeat=args.nfeat, 
                       nhid=nc_config['hidden_channels'], 
                       nclass=args.nclass, 
                       dropout=nc_config['dropout'],
                       lr=nc_config['lr'],
                       weight_decay=nc_config['wd'],
                       device=args.device).to(args.device)

    initial_victim_state_dict = deepcopy(victim_model.state_dict())


    victim_model.fit(features, adj, labels, idx_train, idx_val, train_iters=nc_config['num_epochs'])  
    model_state_dict = victim_model.state_dict()
    victim_model.load_state_dict(model_state_dict)

    surrogate_model = GCN(nfeat=args.nfeat, 
                          nhid=nc_config['hidden_channels'], 
                          nclass=args.nclass, 
                          dropout=nc_config['dropout'],
                          lr=nc_config['lr'],
                          weight_decay=nc_config['wd'],
                          with_relu=False,
                          device=args.device).to(args.device)

    surrogate_model.fit(features, adj, labels, idx_train)
    
   
    ###############      Attack      ###############
    args.n_perturbations = int(adj.sum()/2*args.attack_rate/100)

    attacker = attack_model.init(args=args, 
                                 victim_model=victim_model, 
                                 surrogate_model=surrogate_model, 
                                 nnodes=adj.shape[0])
    
    attacker = attacker.to(args.device)

    attack_model.attack(attacker, args, features.copy(), adj.copy(), labels, idx_train, idx_unlabeled) 
    torch.save(attacker.modified_adj, path_attacked_adj)
    print('Attack Graph Saved')
