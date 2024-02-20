# HideNSeek
Source code of KDD'24 submission
"On Measuring Unnoticeability of Graph Adversarial Attacks: Observations, New Measure, and Applications"

## Environments
- python 3.7.11
- numpy==1.21.2
- torch==1.10.0 (with CUDA 11.3)
- sklearn==1.0.2
- scipy==1.7.3

## Generate Attack Graph

example: cora dataset  /  attack method: PGD attack  /  attack rate: 10%

python attack.py --graph_name cora --attacker pgd --attack_rate 10 --seed 0

python attack.py --graph_name cora --attacker pgd --attack_rate 10 --seed 1

python attack.py --graph_name cora --attacker pgd --attack_rate 10 --seed 2

python attack.py --graph_name cora --attacker pgd --attack_rate 10 --seed 3

python attack.py --graph_name cora --attacker pgd --attack_rate 10 --seed 4

## Measure Noticeability Score by HideNSeek (Q1)

example: cora dataset  /  attack method: PGD attack  /  attack rate: 10%

python q1_main.py --graph_name cora --attacker pgd --attack_rate 10 --seed 0


## Noticeability score curve according to the node classification accuracy (Q2)

example: cora dataset  /  attack method: PGD attack  /  attack rate: 10%  /  average over seed 0 ~ 4

python q2_main.py --graph_name cora --attacker pgd --attack_rate 10 --seed 0

python q2_main.py --graph_name cora --attacker pgd --attack_rate 10 --seed 1

python q2_main.py --graph_name cora --attacker pgd --attack_rate 10 --seed 2

python q2_main.py --graph_name cora --attacker pgd --attack_rate 10 --seed 3

python q2_main.py --graph_name cora --attacker pgd --attack_rate 10 --seed 4

python q2_plot.py --graph_name cora --attacker pgd --attack_rate 10

## Further Application (Q3)

example: cora dataset  /  attack method: PGD attack  /  attack rate: 10%  /  GCN type: MedianGCN

python q3_main.py --graph_name cora --attacker pgd --attack_rate 10 --seed 0 --gcn_name mediangcn
