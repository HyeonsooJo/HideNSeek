import numpy as np
import argparse
import warnings
import random
import torch
import csv
import os
import itertools
import pdb
from attack import *
from utils import *
from leo_train import *
from model import *

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
warnings.filterwarnings(action='ignore')


def HideNSeek(clean_adj, attack_adj, features, args):
    ###############        SEED        ###############
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(args.seed)    
    np.random.seed(args.seed)
    random.seed(args.seed)
    ##################################################

    train_data, valid_data = load_split_data(root=args.root, 
                                                graph_name=args.graph_name, 
                                                attack_info=args.attack_info,
                                                seed=args.seed,
                                                adj=attack_adj, 
                                                features=features)
    
    inputs, train_loader, valid_loader, test_loader = load_lp_dataset(train_data=train_data,
                                                                      valid_data=valid_data, 
                                                                      adj=clean_adj,
                                                                      attack_adj=attack_adj,
                                                                      features=features)
        
    leo_model, input_list = leo_model_init(inputs, args, attack_adj=attack_adj)
    leo_config = json.load(open(os.path.join('config', 'leo', f'leo_{args.graph_name}.json')))
    score = leo_train(args, leo_model, input_list, args.seed, leo_config, 
                                        train_loader, valid_loader, test_loader)

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='cpu if -1')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--root', type=str, default=os.path.join(os.getcwd(), 'data'), help='save data path')

    parser.add_argument('--graph_name', type=str, default='cora', help='graph name: cora, citeeer,...')
    parser.add_argument('--attacker', type=str, default='random', help='attacker model name: random, dice, pgd, ...')
    parser.add_argument('--attack_rate', type=int, default=10, help='percentage of perturbing edges')

    
    args = parser.parse_args()
    args.device = 'cpu' if args.gpu==-1 else 'cuda'

    nc_config = json.load(open(os.path.join('config', 'gcn_nc.json')))
    clean_adj, features, labels, idx_train, idx_val, idx_test, idx_unlabeled = load_dataset(root=args.root, graph_name=args.graph_name)
    args.nfeat = features.shape[1]
    args.nclass = labels.max().item() + 1

    if args.graph_name == 'lastfmasia': 
        args.binary = False 
        args.mse_flag = True
    else: 
        args.binary = True
        args.mse_flag = False
    args.hetero = True if args.graph_name == 'chameleon' or args.graph_name == 'squirrel' else False

    attack_info = f'attacked_adj_{args.attacker}_{args.attack_rate}'
    args.attack_info = attack_info
    path_attacked_adj =  f'{args.root}/{args.graph_name}/attacked_graph/{attack_info}_seed{args.seed}.pt'
    attack_adj = torch.load(path_attacked_adj)

    score = HideNSeek(clean_adj, attack_adj, features, args)
    print(f'Graph: {args.graph_name}  Attack: {args.attacker}  Attack Rate: {args.attack_rate}  Noticeability Score: {score:.2f}')
