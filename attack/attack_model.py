import numpy as np
from .random_attack import Random
from .dice import DICE
from .pgdattack import PGDAttack
from .metattack import Metattack
from .structack import Structack

def init(args, victim_model=None, surrogate_model=None, nnodes=None):
    if args.attacker == "random": return Random(device=args.device)
    elif args.attacker == "dice": return DICE(device=args.device)
    elif args.attacker == "pgd":  return PGDAttack(model=victim_model, nnodes=nnodes, loss_type='CE', device=args.device)
    elif args.attacker == "mettack": return Metattack(model=surrogate_model, nnodes=nnodes, feature_shape=args.nfeat,
                                                       device=args.device, lambda_=0.0, train_iters=100, lr=0.01, with_bias=True).to(args.device)
    elif args.attacker == "metattack": return Metattack(model=surrogate_model, nnodes=nnodes, feature_shape=args.nfeat,
                                                       device=args.device, lambda_=0.0, train_iters=100, lr=0.01, with_bias=True).to(args.device)
    elif args.attacker == "structack": return Structack(device=args.device).to(args.device)
    else: raise NotImplementedError

def attack(attacker, args, ori_features, ori_adj, labels, idx_train, idx_unlabeled):
    # ori_adj: CSR sparse, ori_features: CSR sparse    
    if args.attacker == "random":
        attacker.attack(ori_features, ori_adj, args.n_perturbations,)
        
    elif args.attacker == "dice": 
        attacker.attack(ori_features, ori_adj, labels, args.n_perturbations,)
        
    elif args.attacker == "pgd": 
        attacker.attack(ori_features, ori_adj, labels, np.union1d(idx_train,idx_unlabeled), args.n_perturbations, args=args,)
    
    elif args.attacker == "metattack":        
        attacker.attack(ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations=args.n_perturbations, args=args,)
        
    elif args.attacker == "structack":
        attacker.attack(ori_features, ori_adj, n_perturbations=args.n_perturbations, dataset=args.graph_name,)
    else:
        raise NotImplementedError