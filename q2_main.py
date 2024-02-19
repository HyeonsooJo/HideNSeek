from copy import deepcopy

from scipy.sparse import csr_matrix
import numpy as np
import argparse
import random
import torch
import tqdm
import os

from attack import *
from leo_train import *
from model import *
from utils import *

import warnings
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
warnings.filterwarnings("ignore")


def normalized_adj_th(adj_th):
    mx = adj_th + torch.eye(adj_th.shape[0]).to(adj_th.device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def update_flag(src, dst, adj, th, tmp_train_pos_flag, tmp_train_neg_flag, tmp_valid_pos_flag, tmp_valid_neg_flag):
    num_nodes = adj.shape[0]
    if adj[src, dst] == 1: 
        if tmp_train_pos_flag[src, dst] == 1:
            tmp_train_pos_flag[src, dst] = 0
            tmp_train_pos_flag.eliminate_zeros()
            rm_src, rm_dst = tmp_train_neg_flag.tocoo().row[-1], tmp_train_neg_flag.tocoo().col[-1]
            tmp_train_neg_flag[rm_src, rm_dst] = 0
            tmp_train_neg_flag.eliminate_zeros()

        elif tmp_valid_pos_flag[src, dst] == 1:
            tmp_valid_pos_flag[src, dst] = 0
            tmp_valid_pos_flag.eliminate_zeros()
            rm_src, rm_dst = tmp_valid_neg_flag.tocoo().row[-1], tmp_valid_neg_flag.tocoo().col[-1]
            tmp_valid_neg_flag[rm_src, rm_dst] = 0
            tmp_valid_neg_flag.eliminate_zeros()
        else:
            print('ERROR')
            exit()
    else: 
        # insert, positive sample
        if th < 0.8:
            tmp_train_pos_flag[src, dst] = 1
            while True:
                neg_src, neg_dst = np.random.randint(num_nodes, size=(2))
                if neg_src == neg_dst: continue
                if neg_src > neg_dst:
                    tmp = neg_src
                    neg_src = neg_dst
                    neg_dst = tmp
                if (tmp_train_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_train_neg_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_neg_flag[neg_src, neg_dst] == 0): break
            tmp_train_neg_flag[neg_src, neg_dst] = 1
        else:
            tmp_valid_pos_flag[src, dst] = 1
            while True:
                neg_src, neg_dst = np.random.randint(num_nodes, size=(2))
                if neg_src == neg_dst: continue
                if neg_src > neg_dst:
                    tmp = neg_src
                    neg_src = neg_dst
                    neg_dst = tmp
                if (tmp_train_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_train_neg_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_neg_flag[neg_src, neg_dst] == 0): break
            tmp_valid_neg_flag[neg_src, neg_dst] = 1    
        if tmp_train_neg_flag[src, dst] == 1:
            tmp_train_neg_flag[src, dst] = 0
            tmp_train_neg_flag.eliminate_zeros()
            while True:
                neg_src, neg_dst = np.random.randint(num_nodes, size=(2))
                if neg_src == neg_dst: continue
                if neg_src > neg_dst:
                    tmp = neg_src
                    neg_src = neg_dst
                    neg_dst = tmp
                if (tmp_train_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_train_neg_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_neg_flag[neg_src, neg_dst] == 0): break
            tmp_train_neg_flag[neg_src, neg_dst] = 1
        elif tmp_valid_neg_flag[src, dst] == 1:
            tmp_valid_neg_flag[src, dst] = 0
            tmp_valid_neg_flag.eliminate_zeros()
            while True:
                neg_src, neg_dst = np.random.randint(num_nodes, size=(2))
                if neg_src == neg_dst: continue
                if neg_src > neg_dst:
                    tmp = neg_src
                    neg_src = neg_dst
                    neg_dst = tmp
                if (tmp_train_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_train_neg_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_pos_flag[neg_src, neg_dst] == 0) and \
                (tmp_valid_neg_flag[neg_src, neg_dst] == 0): break
            tmp_valid_neg_flag[neg_src, neg_dst] = 1
    assert tmp_train_pos_flag.sum() == tmp_train_neg_flag.sum()
    assert tmp_valid_pos_flag.sum() == tmp_valid_neg_flag.sum()
    return tmp_train_pos_flag, tmp_train_neg_flag, tmp_valid_pos_flag, tmp_valid_neg_flag

if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='Comparing a sparse matrix with 0 using == is inefficient, try using != instead.')
    ###############     Arguments     ###############
    parser = argparse.ArgumentParser()   
    parser.add_argument('--gpu', type=int, default=0, help='cpu if -1, else gpu')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--root', type=str, default=os.path.join(os.getcwd(), 'data'), help='save data path')


    parser.add_argument('--graph_name', type=str, default='cora', help='graph name: cora, citeeer,...')
    parser.add_argument('--attacker', type=str, default='random', help='attacker model name: random, dice, pgd, ...')
    parser.add_argument('--attack_rate', type=int, default=10, help='percentage of perturbing edges')
    parser.add_argument('--num_point', type=int, default=50, help='')
        
    args = parser.parse_args()
    args.device = 'cpu' if args.gpu==-1 else 'cuda'

    adj, features, labels, idx_train, idx_val, idx_test, idx_unlabeled = load_dataset(root=args.root, graph_name=args.graph_name)
    
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


    ###############      Attack      ###############
    os.makedirs(f'{args.root}/{args.graph_name}/q2_log', exist_ok=True)
    attack_info = f'attacked_adj_{args.attacker}_{args.attack_rate}'
    path_attacked_adj =  f'{args.root}/{args.graph_name}/attacked_graph/{attack_info}_seed{args.seed}.pt'
    path_q2_log =  f'{args.root}/{args.graph_name}/q2_log/{attack_info}_seed{args.seed}.pt'
    attack_adj = torch.load(path_attacked_adj)
    
    q2_log_dict = dict()

    leo_gnn_config = json.load(open(os.path.join('config', 'leo_gnn', f'leo_gnn_{args.graph_name}.json')))
    leo_struct_config = json.load(open(os.path.join('config', 'leo_struct', f'leo_struct_{args.graph_name}.json')))
    leo_config = json.load(open(os.path.join('config', 'leo', f'leo_{args.graph_name}.json')))

    edge_list = np.array((attack_adj - adj).nonzero()).T.tolist()
    edge_list = [(n1, n2) for [n1, n2] in edge_list  if n1 < n2]
    set_seed(args.seed)
    random.shuffle(edge_list)
    
    q2_log_dict['edge_list'] = edge_list

    victim_model = GCN(nfeat=args.nfeat, nhid=nc_config['hidden_channels'], nclass=args.nclass, 
                       dropout=nc_config['dropout'],lr=nc_config['lr'],weight_decay=nc_config['wd'], device=args.device).to(args.device)  
    set_seed(args.seed)
    victim_model.fit(features, adj, labels, idx_train, idx_val, train_iters=nc_config['num_epochs'])
    victim_model.eval()

    feat_np = features.toarray()
    feat_sum = feat_np.sum(axis=1, keepdims=True)
    feat_sum[feat_sum == 0] = 1
    norm_feat_np = feat_np / feat_sum
    feat_th = torch.Tensor(feat_np).to(args.device)
    norm_feat_th = torch.Tensor(norm_feat_np).to(args.device)

    ori_adj_th = torch.Tensor(np.array(adj.toarray())).to(args.device)

    output = victim_model.predict(norm_features=norm_feat_th, norm_adj=normalized_adj_th(ori_adj_th)).detach()
    origin_acc = utils.accuracy(output[idx_test], labels[idx_test]).item()
    origin_unnot = 0.0

    modified_adj_th = torch.Tensor(np.array(adj.toarray())).to(args.device)

    q2_log_dict['random_acc'] = [origin_acc] 
    for (n1,n2)  in tqdm.tqdm(edge_list, leave=False):
        modified_adj_th[n1, n2] = 1-modified_adj_th[n1, n2]
        modified_adj_th[n2, n1] = 1-modified_adj_th[n2, n1]

        output = victim_model.predict(norm_features=norm_feat_th, 
                                        norm_adj=normalized_adj_th(modified_adj_th)).detach()
        q2_log_dict['random_acc'].append(utils.accuracy(output[idx_test], labels[idx_test]).item())
    print('random acc done')


    set_seed(args.seed)
    leo_gnn_model = leo_gnn(num_features=args.nfeat, 
                      hidden_channels=leo_gnn_config['hidden_channels'], 
                      dropout=leo_gnn_config['dropout']).to(args.device)           
    struct_model = leo_gnn(num_features=args.nfeat, 
                      hidden_channels=leo_gnn_config['hidden_channels'], 
                      dropout=leo_gnn_config['dropout']).to(args.device) 
    leo_struct_model = leo_struct_gnn(leo_struct_config, struct_model, args.nfeat, args.device, binary=args.binary, mse_flag=args.mse_flag).to(args.device)
    
    leo_proximity_model = leo_proximity(feat_th, args.device,  hetero=args.hetero, attack_adj=attack_adj) 

    model_list = [leo_gnn_model, leo_struct_model, leo_proximity_model]
    em_model = leo(model_list, 
                   norm_feat_th.shape[-1],
                   leo_gnn_config['hidden_channels'],  
                   leo_config['dropout'], 
                   leo_config['aux_loss_weight'], 
                   args.device).to(args.device)
    

    set_seed(args.seed)
    leo_struct_inputs = [feat_th]

    struct_optim = torch.optim.Adam(params=em_model.leo_struct_model.parameters(), lr=leo_config['lr'], weight_decay=leo_config['wd'])   
    em_model.leo_struct_model.train_dae(struct_optim, leo_struct_inputs)
    em_init_sate_dict = deepcopy(em_model.state_dict())

    pos_edge = np.stack(adj.nonzero()).T
    pos_edge = pos_edge[pos_edge[:, 0] < pos_edge[:, 1]]

    num_nodes = adj.shape[0]
    num_edges = pos_edge.shape[0]

    sample_flag = dict()
    for i in range(num_edges):
        src, dst = pos_edge[i]
        if src not in sample_flag: sample_flag[src] = []
        sample_flag[src].append(dst)
    
    # Negative sample
    set_seed(args.seed)
    neg_edge = np.zeros([num_edges, 2])
    neg_cnt = 0
    while True:
        src, dst = np.random.randint(num_nodes, size=(2))
        if src == dst: continue
        if src > dst:
            tmp = src
            src = dst
            dst = tmp
        if src in sample_flag:
            if dst in sample_flag[src]: continue
        neg_edge[neg_cnt, 0] = src
        neg_edge[neg_cnt, 1] = dst
        if src not in sample_flag: sample_flag[src] = []
        sample_flag[src].append(dst)
        neg_cnt += 1
        if neg_cnt == num_edges: break

    set_seed(args.seed)
    num_train = int(num_edges * 0.8)
    train_mask = np.zeros(num_edges, dtype=np.bool)
    train_mask[np.random.choice(np.arange(num_edges), num_train, replace=False)] = True

    origin_train_pos = pos_edge[train_mask]
    origin_valid_pos = pos_edge[~train_mask]
    origin_train_neg = neg_edge[train_mask]
    origin_valid_neg = neg_edge[~train_mask]
    
    tmp_train_pos_flag = csr_matrix(
        (np.ones(origin_train_pos.shape[0]), 
         (origin_train_pos[:, 0], origin_train_pos[:, 1])), 
        (num_nodes, num_nodes))
    tmp_valid_pos_flag = csr_matrix(
        (np.ones(origin_valid_pos.shape[0]), 
         (origin_valid_pos[:, 0], origin_valid_pos[:, 1])), 
        (num_nodes, num_nodes))
    tmp_train_neg_flag = csr_matrix(
        (np.ones(origin_train_neg.shape[0]), 
         (origin_train_neg[:, 0], origin_train_neg[:, 1])), 
        (num_nodes, num_nodes))
    tmp_valid_neg_flag = csr_matrix(
        (np.ones(origin_valid_neg.shape[0]), 
         (origin_valid_neg[:, 0], origin_valid_neg[:, 1])), 
        (num_nodes, num_nodes))
    
    tmp_norm_adj_th = torch.zeros(adj.shape).to(args.device)
    diag = torch.eye(tmp_norm_adj_th.shape[0]).to(args.device)
    interval = int(len(edge_list) / args.num_point)
    if interval < 5: interval = 5

    set_seed(args.seed)
    random_threshold = np.random.random(len(edge_list))
    
    tmp_attack_add_edge = []
    q2_log_dict['random_score'] = [origin_unnot] 
    for i in tqdm.tqdm(range(len(edge_list)), leave=False):                    
        src, dst = edge_list[i]
        if adj[src, dst] == 0: tmp_attack_add_edge.append([src, dst])
        th = random_threshold[i]
        tmp_train_pos_flag, tmp_train_neg_flag, \
        tmp_valid_pos_flag, tmp_valid_neg_flag = update_flag(src, dst, adj, th, \
                                                                tmp_train_pos_flag, \
                                                                tmp_train_neg_flag, \
                                                                tmp_valid_pos_flag, \
                                                                tmp_valid_neg_flag)

        if (i+1) % interval == 0 or i == len(edge_list) - 1:
            set_seed(args.seed)
            tmp_train_pos_row, tmp_train_pos_col = tmp_train_pos_flag.tocoo().row, tmp_train_pos_flag.tocoo().col
            tmp_train_neg_row, tmp_train_neg_col = tmp_train_neg_flag.tocoo().row, tmp_train_neg_flag.tocoo().col
            tmp_valid_pos_row, tmp_valid_pos_col = tmp_valid_pos_flag.tocoo().row, tmp_valid_pos_flag.tocoo().col
            tmp_valid_neg_row, tmp_valid_neg_col = tmp_valid_neg_flag.tocoo().row, tmp_valid_neg_flag.tocoo().col
            
            tmp_train_pos_sample = torch.vstack([torch.LongTensor(tmp_train_pos_row), torch.LongTensor(tmp_train_pos_col)]).T
            tmp_train_neg_sample = torch.vstack([torch.LongTensor(tmp_train_neg_row), torch.LongTensor(tmp_train_neg_col)]).T
            tmp_train_sample = torch.vstack([tmp_train_pos_sample, tmp_train_neg_sample]).to(args.device)
            tmp_train_label = torch.hstack([torch.ones(tmp_train_pos_sample.shape[0]), torch.zeros(tmp_train_neg_sample.shape[0])]).to(args.device)
            
            tmp_valid_pos_sample = torch.vstack([torch.LongTensor(tmp_valid_pos_row), torch.LongTensor(tmp_valid_pos_col)]).T
            tmp_valid_neg_sample = torch.vstack([torch.LongTensor(tmp_valid_neg_row), torch.LongTensor(tmp_valid_neg_col)]).T
            tmp_valid_sample = torch.vstack([tmp_valid_pos_sample, tmp_valid_neg_sample]).to(args.device)
            tmp_valid_label = torch.hstack([torch.ones(tmp_valid_pos_sample.shape[0]), torch.zeros(tmp_valid_neg_sample.shape[0])]).to(args.device)
            
            tmp_norm_adj_th[:] = 0
            tmp_norm_adj_th[tmp_train_pos_row, tmp_train_pos_col] = 1.0
            tmp_norm_adj_th[tmp_train_pos_col, tmp_train_pos_row] = 1.0
            tmp_norm_adj_th = tmp_norm_adj_th + diag
            row_sum = tmp_norm_adj_th.sum(dim=1, keepdims=True)
            r_inv = row_sum.pow(-1/2)
            r_inv[torch.isinf(r_inv)] = 0.
            tmp_norm_adj_th = (tmp_norm_adj_th * r_inv) * r_inv.T
        
            leo_gnn_input = [tmp_norm_adj_th, norm_feat_th]
            em_input = [leo_gnn_input, leo_struct_inputs, None]

            em_model.load_state_dict(em_init_sate_dict)
            optimizer = torch.optim.Adam(params=em_model.parameters(), lr=leo_config['lr'], weight_decay=leo_config['wd'])  
            best_val_auroc = 0
            with tqdm.tqdm(range(leo_config['num_epochs'] ), leave=False, file=sys.stderr) as pbar:
                for epoch in pbar :
                    pbar.set_description(f'Epoch - {epoch}')
                    em_model.train()
                    out = em_model(em_input, tmp_train_sample, tmp_train_label)
                    y=tmp_train_label.view(-1)
                    if not leo_config['filter_rate'] == 0.0:
                        out_idx = out.detach()[y==1].sort()[1]>int(out.shape[0]*leo_config['filter_rate'])
                        out=torch.cat((out[y==0], out[y==1][out_idx]))
                        y=torch.cat((y[y==0], y[y==1][out_idx]))
                    
                    loss = em_model._loss(out, y.view(-1)) + em_model.auxloss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    em_model.eval()
                    out = em_model(em_input, tmp_valid_sample, tmp_valid_label)
                    val_auroc = roc_auc_score(tmp_valid_label.cpu().detach(), out.cpu().detach())
                    pbar.set_postfix(loss = loss, acc = np.round(val_auroc, 3))
                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        test_sample = torch.vstack([torch.LongTensor(pos_edge), torch.LongTensor(np.array(tmp_attack_add_edge))]).to(args.device)
                        test_label = torch.hstack([torch.ones(pos_edge.shape[0]), torch.zeros(len(tmp_attack_add_edge))]).to(args.device)
                        out = em_model(em_input, test_sample, test_label)
                        test_auroc = roc_auc_score(test_label.cpu().detach(), out.cpu().detach())
                        best_model_state_dict = deepcopy(em_model.state_dict()) 
            score = test_auroc
        else:
            score = -1
        q2_log_dict['random_score'].append(score)

    print('random score done')
                                      
    
    ########### top 1 #############
    tmp_train_pos_flag = csr_matrix(
        (np.ones(origin_train_pos.shape[0]), 
         (origin_train_pos[:, 0], origin_train_pos[:, 1])), 
        (num_nodes, num_nodes))
    tmp_valid_pos_flag = csr_matrix(
        (np.ones(origin_valid_pos.shape[0]), 
         (origin_valid_pos[:, 0], origin_valid_pos[:, 1])), 
        (num_nodes, num_nodes))
    tmp_train_neg_flag = csr_matrix(
        (np.ones(origin_train_neg.shape[0]), 
         (origin_train_neg[:, 0], origin_train_neg[:, 1])), 
        (num_nodes, num_nodes))
    tmp_valid_neg_flag = csr_matrix(
        (np.ones(origin_valid_neg.shape[0]), 
         (origin_valid_neg[:, 0], origin_valid_neg[:, 1])), 
        (num_nodes, num_nodes))
    

    modified_adj_th = torch.Tensor(np.array(adj.toarray())).to(args.device)
    q2_log_dict['greedy_score'] = []
    q2_log_dict['greedy_score'].append(((-1, -1), origin_unnot, origin_acc))  
    victim_model.eval()
    

    tmp_norm_adj_th[:] = 0
    tmp_norm_adj_th[origin_train_pos[:, 0], origin_train_pos[:, 1]] = 1.0
    tmp_norm_adj_th[origin_train_pos[:, 1], origin_train_pos[:, 0]] = 1.0
    tmp_norm_adj_th = tmp_norm_adj_th + diag
    row_sum = tmp_norm_adj_th.sum(dim=1, keepdims=True)
    r_inv = row_sum.pow(-1/2)
    r_inv[torch.isinf(r_inv)] = 0.
    tmp_norm_adj_th = (tmp_norm_adj_th * r_inv) * r_inv.T
    leo_gnn_input = [tmp_norm_adj_th, norm_feat_th]
    em_input = [leo_gnn_input, leo_struct_inputs, None]

    set_seed(args.seed)
    em_model.load_state_dict(em_init_sate_dict)
    
    optimizer = torch.optim.Adam(params=em_model.parameters(), lr=leo_config['lr'], weight_decay=leo_config['wd'])  
    best_val_auroc = 0

    with tqdm.tqdm(range(leo_config['num_epochs']), leave=False, file=sys.stderr) as pbar:
        origin_train_sample = torch.LongTensor(np.vstack([origin_train_pos, origin_train_neg])).to(args.device)
        origin_train_label = torch.hstack([torch.ones(origin_train_pos.shape[0]), torch.zeros(origin_train_neg.shape[0])]).to(args.device)
        
        origin_valid_sample = torch.LongTensor(np.vstack([origin_valid_pos, origin_valid_neg])).to(args.device)
        origin_valid_label = torch.hstack([torch.ones(origin_valid_pos.shape[0]), torch.zeros(origin_valid_neg.shape[0])]).to(args.device)
        
        for epoch in pbar :
            pbar.set_description(f'Epoch - {epoch}')
            em_model.train()
            out = em_model(em_input, origin_train_sample, origin_train_label)
            y=origin_train_label.view(-1)
            if not leo_config['filter_rate'] == 0.0:
                out_idx = out.detach()[y==1].sort()[1]>int(out.shape[0]*leo_config['filter_rate'])
                out=torch.cat((out[y==0], out[y==1][out_idx]))
                y=torch.cat((y[y==0], y[y==1][out_idx]))
            
            loss = em_model._loss(out, y) + em_model.auxloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            em_model.eval()
            out = em_model(em_input, origin_valid_sample, origin_valid_label)
            val_auroc = roc_auc_score(origin_valid_label.cpu().detach(), out.cpu().detach())
            pbar.set_postfix(loss = loss, acc = np.round(val_auroc,3))
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_model_state_dict = deepcopy(em_model.state_dict()) 


    em_model.load_state_dict(best_model_state_dict)
    em_model.eval()
    out = em_model(em_input, torch.LongTensor(edge_list).to(args.device))
    insert_mask = np.array([adj[src, dst] == 0 for src, dst in edge_list])

    score = np.zeros(len(edge_list))
    score[insert_mask] = out[insert_mask].detach().cpu().numpy()
    score[~insert_mask] = 1 - out[~insert_mask].detach().cpu().numpy()

    order = np.argsort(-score)
    greedy_edge_list = [edge_list[order[i]] for i in range(len(edge_list))]

    tmp_attack_add_edge = []
    modified_adj_th = torch.Tensor(np.array(adj.copy().todense())).to(args.device)                
    for i in tqdm.tqdm(range(len(greedy_edge_list)), leave=False):        
        src, dst = greedy_edge_list[i]
        if adj[src, dst] == 0: tmp_attack_add_edge.append([src, dst])
        th = random_threshold[i]
        tmp_train_pos_flag, tmp_train_neg_flag, \
        tmp_valid_pos_flag, tmp_valid_neg_flag = update_flag(src, dst, adj, th, \
                                                                tmp_train_pos_flag, \
                                                                tmp_train_neg_flag, \
                                                                tmp_valid_pos_flag, \
                                                                tmp_valid_neg_flag)
        
        modified_adj_th[src, dst] = 1-modified_adj_th[src, dst]
        modified_adj_th[dst, src] = 1-modified_adj_th[dst, src]   
        output = victim_model.predict(norm_features=norm_feat_th, 
                                    norm_adj=normalized_adj_th(modified_adj_th)).detach()
        acc_evasion = utils.accuracy(output[idx_test], labels[idx_test]).item()
        
        ######## train unnot model for post performance ##########
        if (i+1) % interval == 0 or i == len(edge_list) - 1:
            set_seed(args.seed)
            tmp_train_pos_row, tmp_train_pos_col = tmp_train_pos_flag.tocoo().row, tmp_train_pos_flag.tocoo().col
            tmp_train_neg_row, tmp_train_neg_col = tmp_train_neg_flag.tocoo().row, tmp_train_neg_flag.tocoo().col
            tmp_valid_pos_row, tmp_valid_pos_col = tmp_valid_pos_flag.tocoo().row, tmp_valid_pos_flag.tocoo().col
            tmp_valid_neg_row, tmp_valid_neg_col = tmp_valid_neg_flag.tocoo().row, tmp_valid_neg_flag.tocoo().col
            
            tmp_train_pos_sample = torch.vstack([torch.LongTensor(tmp_train_pos_row), torch.LongTensor(tmp_train_pos_col)]).T
            tmp_train_neg_sample = torch.vstack([torch.LongTensor(tmp_train_neg_row), torch.LongTensor(tmp_train_neg_col)]).T
            tmp_train_sample = torch.vstack([tmp_train_pos_sample, tmp_train_neg_sample]).to(args.device)
            tmp_train_label = torch.hstack([torch.ones(tmp_train_pos_sample.shape[0]), torch.zeros(tmp_train_neg_sample.shape[0])]).to(args.device)
            
            tmp_valid_pos_sample = torch.vstack([torch.LongTensor(tmp_valid_pos_row), torch.LongTensor(tmp_valid_pos_col)]).T
            tmp_valid_neg_sample = torch.vstack([torch.LongTensor(tmp_valid_neg_row), torch.LongTensor(tmp_valid_neg_col)]).T
            tmp_valid_sample = torch.vstack([tmp_valid_pos_sample, tmp_valid_neg_sample]).to(args.device)
            tmp_valid_label = torch.hstack([torch.ones(tmp_valid_pos_sample.shape[0]), torch.zeros(tmp_valid_neg_sample.shape[0])]).to(args.device)
            
            tmp_norm_adj_th[:] = 0
            tmp_norm_adj_th[tmp_train_pos_row, tmp_train_pos_col] = 1.0
            tmp_norm_adj_th[tmp_train_pos_col, tmp_train_pos_row] = 1.0
            tmp_norm_adj_th = tmp_norm_adj_th + diag
            row_sum = tmp_norm_adj_th.sum(dim=1, keepdims=True)
            r_inv = row_sum.pow(-1/2)
            r_inv[torch.isinf(r_inv)] = 0.
            tmp_norm_adj_th = (tmp_norm_adj_th * r_inv) * r_inv.T
        
            leo_gnn_input = [tmp_norm_adj_th, norm_feat_th]
            em_input = [leo_gnn_input, leo_struct_inputs, None]

            em_model.load_state_dict(em_init_sate_dict)
            optimizer = torch.optim.Adam(params=em_model.parameters(), lr=leo_config['lr'], weight_decay=leo_config['wd'])  
            best_val_auroc = 0
            with tqdm.tqdm(range(leo_config['num_epochs']), leave=False, file=sys.stderr) as pbar:
                for epoch in pbar :
                    pbar.set_description(f'Epoch - {epoch}')
                    em_model.train()
                    out = em_model(em_input, tmp_train_sample, tmp_train_label)
                    y=tmp_train_label.view(-1)
                    if not leo_config['filter_rate'] == 0.0:
                        out_idx = out.detach()[y==1].sort()[1]>int(out.shape[0]*leo_config['filter_rate'])
                        out=torch.cat((out[y==0], out[y==1][out_idx]))
                        y=torch.cat((y[y==0], y[y==1][out_idx]))
                    loss = em_model._loss(out, y) + em_model.auxloss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    em_model.eval()
                    out = em_model(em_input, tmp_valid_sample, tmp_valid_label)
                    val_auroc = roc_auc_score(tmp_valid_label.cpu().detach(), out.cpu().detach())
                    pbar.set_postfix(loss = loss, acc = np.round(val_auroc,3))
                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        test_sample = torch.vstack([torch.LongTensor(pos_edge), torch.LongTensor(np.array(tmp_attack_add_edge))]).to(args.device)
                        test_label = torch.hstack([torch.ones(pos_edge.shape[0]), torch.zeros(len(tmp_attack_add_edge))]).to(args.device)
                        out = em_model(em_input, test_sample, test_label)
                        test_auroc = roc_auc_score(test_label.cpu().detach(), out.cpu().detach())
                        best_model_state_dict = deepcopy(em_model.state_dict()) 
            score = test_auroc
        else:
            score=-1  
        q2_log_dict['greedy_score'].append(((n1, n2), score, acc_evasion))    
    
    print('greedy score done')           
    torch.save(q2_log_dict, path_q2_log)
