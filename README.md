# HideNSeek
Source code of KDD'24 submission
"On Measuring Unnoticeability of Graph Adversarial Attacks: Observations, New Measure, and Applications"

## Generate Attack Graph

example: cora dataset  /  attack method: random  /  attack rate: 10%

python attack.py --graph_name cora --attacker random --attack_rate 10 --seed 0


## Measure Noticeability Score by HideNSeek (Q1)

example: cora dataset  /  attack method: random  /  attack rate: 10%

python q1_main.py --graph_name cora --attacker random --attack_rate 10 --seed 0


## Noticeability score curve according to the node classification accuracy (Q2)

example: cora dataset  /  attack method: random  /  attack rate: 10%  /  average over seed 0 ~ 4

python q2_main.py --graph_name cora --attacker random --attack_rate 10 --seed 0

python q2_main.py --graph_name cora --attacker random --attack_rate 10 --seed 1

python q2_main.py --graph_name cora --attacker random --attack_rate 10 --seed 2

python q2_main.py --graph_name cora --attacker random --attack_rate 10 --seed 3

python q2_main.py --graph_name cora --attacker random --attack_rate 10 --seed 4

python q2_plot.py --graph_name cora --attacker random --attack_rate 10
