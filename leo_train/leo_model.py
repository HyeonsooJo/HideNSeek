import numpy as np
import torch

from utils import *
from model import *


def leo_model_init(inputs, args, attack_adj=None):
    mp_adj, norm_adj, features, norm_features = inputs
    mp_adj, norm_adj, features, norm_features = mp_adj.to(args.device), norm_adj.to(args.device), features.to(args.device), norm_features.to(args.device)

    leo_gnn_config = json.load(open(os.path.join('config', 'leo_gnn', f'leo_gnn_{args.graph_name}.json')))
    leo_struct_config = json.load(open(os.path.join('config', 'leo_struct', f'leo_struct_{args.graph_name}.json')))
    leo_config = json.load(open(os.path.join('config', 'leo', f'leo_{args.graph_name}.json')))

    leo_gnn_model = leo_gnn(num_features=args.nfeat, 
                      hidden_channels=leo_gnn_config['hidden_channels'], 
                      dropout=leo_gnn_config['dropout']).to(args.device) 
    leo_gnn_input = [norm_adj, norm_features]
    
    struct_model = leo_gnn(num_features=args.nfeat, 
                      hidden_channels=leo_gnn_config['hidden_channels'], 
                      dropout=leo_gnn_config['dropout']).to(args.device) 
    leo_struct_model = leo_struct_gnn(leo_struct_config, struct_model, args.nfeat, args.device, binary=args.binary, mse_flag=args.mse_flag).to(args.device)
    leo_struct_inputs = [features]

    leo_proximity_model = leo_proximity(features, args.device,  hetero=args.hetero, attack_adj=attack_adj) 


    model_list = [leo_gnn_model, leo_struct_model, leo_proximity_model]
    input_list = [leo_gnn_input, leo_struct_inputs, None]
    

    em_model = leo(model_list, 
                   inputs[-1].shape[1],
                   leo_gnn_config['hidden_channels'],  
                   leo_config['dropout'], 
                   leo_config['aux_loss_weight'], 
                   args.device).to(args.device)
    
    return em_model, input_list
    
