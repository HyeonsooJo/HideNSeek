from scipy.sparse import csr_matrix
import numpy as np
import argparse
import warnings
import random
import torch
import os

from attack import *
from utils import *
from leo_train import *
from model import *

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
warnings.filterwarnings(action='ignore')


def filtering(clean_adj, attack_adj, features, args, seed):
    attack_adj_th = torch.Tensor(attack_adj.toarray()).to(args.device)
    num_attack = int((attack_adj - clean_adj).nonzero()[0].shape[0] / 2)

    ###############        SEED        ###############
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)
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

    leo_config = json.load(open(os.path.join('config', 'leo', f'leo_{args.graph_name}.json')))
    leo_model, input_list = leo_model_init(inputs, args, attack_adj=attack_adj)
    score = leo_train(args, leo_model, input_list, args.seed, leo_config, 
                                        train_loader, valid_loader, test_loader)
    leo_model.eval()
    edge_index = np.stack(attack_adj.nonzero())
    edge_index = edge_index[:, edge_index[0] < edge_index[1]]
    edge_index = torch.LongTensor(edge_index.T).to(args.device)
    out = leo_model(input_list, edge_index).cpu().detach()

    mask = out.sort()[1][num_attack:]

    filter_edge_index = edge_index[mask, :]
    filter_adj = torch.zeros_like(attack_adj_th)
    filter_adj[filter_edge_index[:, 0], filter_edge_index[:, 1]] = 1.0
    filter_adj[filter_edge_index[:, 1], filter_edge_index[:, 0]] = 1.0
    return filter_adj

def main(gcn_name, clean_adj, attack_adj, features, args, seed):
    nc_config = json.load(open(os.path.join('config', 'gcn_nc.json')))
    
    filter_adj_th = filtering(clean_adj, attack_adj, features, args, seed)
    filter_adj = csr_matrix(filter_adj_th.detach().cpu().numpy())

    if gcn_name == 'gcn':
        victim_model = GCN(nfeat=args.nfeat,
                           nhid=nc_config['hidden_channels'], 
                           nclass=args.nclass, 
                           dropout=nc_config['dropout'],
                           lr=nc_config['lr'],
                           weight_decay=nc_config['wd'],
                           device=args.device).to(args.device)       
    elif gcn_name == 'rgcn':
        victim_model = RGCN(nnodes=clean_adj.shape[0],
                           nfeat=args.nfeat,
                           nhid=nc_config['hidden_channels'], 
                           nclass=args.nclass, 
                           dropout=nc_config['dropout'],
                           lr=nc_config['lr'],
                           weight_decay=nc_config['wd'],
                           device=args.device).to(args.device)  
    elif gcn_name == 'mediangcn':
        victim_model = MedianGCN(
                           nfeat=args.nfeat,
                           nhid=nc_config['hidden_channels'], 
                           nclass=args.nclass, 
                           dropout=nc_config['dropout'],
                           lr=nc_config['lr'],
                           weight_decay=nc_config['wd'],
                           device=args.device).to(args.device)  


    initial_victim_state_dict = deepcopy(victim_model.state_dict())

    # poisoning attack 
    victim_model.load_state_dict(initial_victim_state_dict)
    ###############        SEED        ###############
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)
    ##############        SEED        ###############
    victim_model.fit(features, attack_adj, labels, idx_train, idx_val, train_iters=nc_config['num_epochs'])  

    victim_model.set_data(features, attack_adj, labels)
    output = victim_model.predict().detach()
    attack_acc = utils.accuracy(output[idx_test], labels[idx_test]).item() 


    # filter graph poisoning
    victim_model.load_state_dict(initial_victim_state_dict)
    ###############        SEED        ###############
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)
    ###############        SEED        ###############

    victim_model.fit(features, filter_adj, labels, idx_train, idx_val, train_iters=nc_config['num_epochs'])  
    victim_model.set_data(features, filter_adj, labels)
    output = victim_model.predict().detach()
    filter_acc = utils.accuracy(output[idx_test], labels[idx_test]).item() 

    return attack_acc, filter_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu', type=int, default=0, help='cpu if -1, else gpu')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--root', type=str, default=os.path.join(os.getcwd(), 'data'), help='save data path')

    parser.add_argument('--graph_name', type=str, default='cora', help='graph name: cora, citeeer,...')
    parser.add_argument('--attacker', type=str, default='random', help='attacker model name: random, dice, pgd, ...')
    parser.add_argument('--attack_rate', type=int, default=10, help='percentage of perturbing edges')
    parser.add_argument('--gcn_name', type=str, default='gcn', help='gcn, rgcn, mediangcn')
    
    args = parser.parse_args()
    args.device = 'cpu' if args.gpu==-1 else 'cuda'
    

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

    args.attack_info = f'attacked_adj_{args.attacker}_{args.attack_rate}'
    path_attacked_adj =  f'{args.root}/{args.graph_name}/attacked_graph/{args.attack_info}_seed{args.seed}.pt'
    attack_adj = torch.load(path_attacked_adj)

    print(f'Robustness experiment: Attacker: {args.attacker}  # Attack Rate: {args.attack_rate}')

    attack_acc, filter_acc = main(args.gcn_name, clean_adj, attack_adj, features, args, args.seed)
    print(f'[RESULT] Attack Acc: {attack_acc:.3f}   Filter Acc: {filter_acc:.3f}')
