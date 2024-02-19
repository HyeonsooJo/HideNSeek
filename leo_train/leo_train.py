from sklearn.metrics import roc_auc_score
from copy import deepcopy
import pdb
import torch
import tqdm
import sys

from utils import *


def train_one_epoch(model, inputs, train_loader, optimizer, device, filter_rate=0.0):
    model.train()
    edge_index_list, y_list = [], []
    for edge_index, y in train_loader:
        edge_index_list.append(edge_index)
        y_list.append(y)
    edge_index = torch.cat(edge_index_list)
    y = torch.cat(y_list)
    edge_index, y = edge_index.to(device), y.to(device)
    out = model(inputs, edge_index, y)

    if not filter_rate == 0.0:
        y=y.view(-1)
        out_idx = out.detach()[y==1].sort()[1]>int(out.shape[0]*filter_rate)
        out=torch.cat((out[y==0], out[y==1][out_idx]))
        y=torch.cat((y[y==0], y[y==1][out_idx]))
    loss = model._loss(out, y.view(-1)) + model.auxloss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def auroc(model, inputs, loader, device):
    model.eval()
    edge_index_list, y_list = [], []
    for edge_index, y in loader:
        edge_index_list.append(edge_index)
        y_list.append(y)
    edge_index = torch.cat(edge_index_list)
    y = torch.cat(y_list)
    edge_index, y = edge_index.to(device), y.to(device)
    out = model(inputs, edge_index, y=y)

    return roc_auc_score(y.detach().cpu(), out.detach().cpu())


def train(args, model, inputs, seed, leo_config, train_loader, valid_loader, test_loader):
    ###############        SEED        ###############
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)
    ##################################################
    optimizer = torch.optim.Adam(params=model.parameters(), lr=leo_config['lr'], weight_decay=leo_config['wd'])  
    best_val_auroc = 0
    model.train()

    with tqdm.tqdm(range(leo_config['num_epochs']), leave=True, file=sys.stderr) as pbar :
        for epoch in pbar :

            pbar.set_description(f'Epoch - {epoch}')
            loss = train_one_epoch(model, inputs, train_loader, optimizer, args.device, filter_rate=leo_config['filter_rate'])

            val_auroc = auroc(model, inputs, valid_loader, args.device)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_test_auroc = auroc(model, inputs, test_loader, args.device)
                best_model_state_dict = deepcopy(model.state_dict()) 

            pbar.set_postfix(loss = loss, val_acc = np.round(val_auroc, 3), test_acc = np.round(best_test_auroc, 3))

    return best_test_auroc, best_model_state_dict


def leo_train(args, leo_model, input_list, seed, leo_config, train_loader, valid_loader, test_loader):    
    ###  pretrain only dae #### 
    leo_model.train()

    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)
    optimizer = torch.optim.Adam(params=leo_model.leo_struct_model.parameters(), lr=leo_config['lr'], weight_decay=leo_config['wd'])   
    leo_model.leo_struct_model.train_dae(optimizer, input_list[1])

    ### Whole model training (ensemble) ###
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)

    best_test_auroc, _ = train(args, leo_model, input_list, seed, leo_config, train_loader, valid_loader, test_loader)
    
    return best_test_auroc

