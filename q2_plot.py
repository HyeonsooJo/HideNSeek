import argparse
import torch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.path import Path


plt.style.use('seaborn') 
sns.set_theme(style="whitegrid")

def q2_plot(graph_name='cora', attacker='random', attack_rate=10):

    attack_info = f'attacked_adj_{attacker}_{attack_rate}'
    
    seeds = [0, 1, 2, 3, 4]

    q2_log_dict_list = []
    if (attacker == 'pgd') and (graph_name=='lastfmasia' or graph_name == 'citeseer'): avg_check = 3
    else: avg_check = 5

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,7)) 

    for seed in seeds:
        path_q2_log =  f'data/{graph_name}/q2_log/{attack_info}_seed{seed}.pt'

        q2_log_dict_list.append(torch.load(path_q2_log))
        acc_random, acc_greedy, unnot_random, unnot_greedy, length = [], [], [], [], []
    
    for edge_dict in q2_log_dict_list:
        mask = np.array(edge_dict['random_score']) == -1
        tmp_acc_random = np.array([acc for acc in edge_dict['random_acc']])
        tmp_acc_random[mask] = -1
        tmp_unnot_random = np.array([y for y in edge_dict['random_score']])
        tmp_unnot_random[mask] = -1
        tmp_acc_greedy = np.array([acc for _, _, acc in edge_dict['greedy_score']])
        tmp_unnot_greedy = np.array([y for _, y, _ in edge_dict['greedy_score']])


        tmp_acc_greedy[mask] = -1
        tmp_unnot_greedy[mask] = -1
        acc_random.append(tmp_acc_random)
        acc_greedy.append(tmp_acc_greedy)
        unnot_random.append(tmp_unnot_random)
        unnot_greedy.append(tmp_unnot_greedy)
        length.append(len(tmp_acc_random))

    acc_random_mean, unnot_random_mean = np.zeros(np.max(length)), np.zeros(np.max(length))
    unnot_random_denom = np.zeros(np.max(length))

    for acc, unnot in zip(acc_random, unnot_random):
        assert(len(acc)==len(unnot))
        unnot=np.array(unnot)
        
        acc_random_mean[0:len(acc)][acc!=-1] += acc[acc!=-1]
        unnot_random_mean[0:len(unnot)][unnot!=-1] += unnot[unnot!=-1]

        acc_random_mean[len(acc):] += acc[-1]
        unnot_random_mean[len(unnot):] += unnot[-1]
 
        unnot_random_denom[0:len(unnot)][unnot!=-1] += 1
        unnot_random_denom[len(unnot):] += 1

    unnot_random_mean[unnot_random_denom<avg_check] = -1
    unnot_random_denom[unnot_random_denom<avg_check] = 0
    unnot_random_denom[unnot_random_denom==0] = 1
    acc_random_mean, unnot_random_mean = acc_random_mean/unnot_random_denom*100, unnot_random_mean/unnot_random_denom

    acc_greedy_mean, unnot_greedy_mean = np.zeros(np.max(length)), np.zeros(np.max(length))
    unnot_greedy_denom = np.zeros(np.max(length))

    for acc, unnot in zip(acc_greedy, unnot_greedy):
        assert(len(acc)==len(unnot))
        unnot=np.array(unnot)

        acc_greedy_mean[0:len(acc)][acc!=-1] += acc[acc!=-1]
        unnot_greedy_mean[0:len(unnot)][unnot!=-1] += unnot[unnot!=-1]

        acc_greedy_mean[len(acc):] += acc[-1]
        unnot_greedy_mean[len(unnot):] += unnot[-1]
 
        unnot_greedy_denom[0:len(unnot)][unnot!=-1] += 1
        unnot_greedy_denom[len(unnot):] += 1

    unnot_greedy_mean[unnot_greedy_denom<avg_check] = -1
    unnot_greedy_denom[unnot_greedy_denom<avg_check] = 0
    unnot_greedy_denom[unnot_greedy_denom==0] = 1

    acc_greedy_mean, unnot_greedy_mean = acc_greedy_mean/unnot_greedy_denom*100, unnot_greedy_mean/unnot_greedy_denom

    filtered = unnot_greedy_mean >= 0
    acc_greedy_mean = acc_greedy_mean[filtered]
    unnot_greedy_mean = unnot_greedy_mean[filtered]

    filtered = unnot_random_mean >= 0
    acc_random_mean = acc_random_mean[filtered]
    unnot_random_mean = unnot_random_mean[filtered]

    acc_random_mean_start = acc_random_mean[0]
    acc_greedy_mean_start = acc_greedy_mean[0]
    acc_random_mean[acc_random_mean>acc_random_mean_start] = acc_random_mean_start
    acc_greedy_mean[acc_greedy_mean>acc_greedy_mean_start] = acc_greedy_mean_start

    acc_random_mean_final = acc_random_mean[-1]
    acc_greedy_mean_final = acc_greedy_mean[-1]
    acc_random_mean[acc_random_mean<acc_random_mean_final] = acc_random_mean_final
    acc_greedy_mean[acc_greedy_mean<acc_greedy_mean_final] = acc_greedy_mean_final


    random_verts = [(acc,unnot) for acc, unnot in zip(acc_random_mean, unnot_random_mean)]
    random_codes = [Path.MOVETO, Path.LINETO] + [Path.CURVE3] * (len(random_verts) - 4) + [Path.LINETO, Path.LINETO]
    random_curve_path = Path(random_verts, random_codes)

    greedy_verts = [(acc,unnot) for acc, unnot in zip(acc_greedy_mean, unnot_greedy_mean)]
    greedy_codes = [Path.MOVETO, Path.LINETO] + [Path.CURVE3] * (len(greedy_verts) - 4) + [Path.LINETO, Path.LINETO]
    greedy_curve_path = Path(greedy_verts, greedy_codes)


    random_verts = [(acc_random_mean[-1], unnot_random_mean[0])]
    for acc, unnot in zip(acc_random_mean, unnot_random_mean): random_verts.append((acc,unnot))
    random_verts.append((acc_random_mean[-1], unnot_random_mean[0]))
    random_codes = [Path.MOVETO, Path.LINETO, Path.LINETO] + [Path.CURVE3] * (len(random_verts) - 6) + [Path.LINETO, Path.LINETO, Path.LINETO]
    random_area_path = Path(random_verts, random_codes)

    greedy_verts = [(acc_greedy_mean[-1], unnot_greedy_mean[0])]
    for acc, unnot in zip(acc_greedy_mean, unnot_greedy_mean): greedy_verts.append((acc,unnot))
    greedy_verts.append((acc_greedy_mean[-1], unnot_greedy_mean[0]))
    greedy_codes = [Path.MOVETO, Path.LINETO, Path.LINETO] + [Path.CURVE3] * (len(greedy_verts) - 6) + [Path.LINETO, Path.LINETO, Path.LINETO]
    greedy_area_path = Path(greedy_verts, greedy_codes)


    random_area_color = '#dee4f0'
    greedy_area_color = '#a9c7e0'

    random_curve_color = '#a9c7e0'
    greedy_curve_color = '#045a8d'

    axes.add_patch(patches.PathPatch(random_area_path, facecolor=random_area_color, edgecolor='none', alpha=1.0, linewidth=2, fill=True))
    axes.add_patch(patches.PathPatch(random_curve_path, facecolor='none',edgecolor=random_curve_color, alpha=1.0,  lw=2))    
    axes.add_patch(patches.PathPatch(greedy_area_path, facecolor = greedy_area_color, edgecolor='none', alpha=1.0, linewidth=2, fill=True))
    axes.add_patch(patches.PathPatch(greedy_area_path, facecolor = 'none', edgecolor='black', alpha=0.5, linewidth=0.3, fill=True, hatch='//'))
    axes.add_patch(patches.PathPatch(greedy_curve_path, facecolor='none',edgecolor=greedy_curve_color, alpha=1.0,  lw=2))

    axes.set_xlabel('Node Classification accuracy (%)', fontsize=30)
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axes.set_ylabel('HideNSeek', fontsize=30)

    axes.tick_params(labelsize=20)
    fig.set_tight_layout(True)
    eps = 0.01

    min_x = np.minimum(acc_greedy_mean.min(), acc_random_mean.min())
    max_x = np.maximum(acc_greedy_mean.max(), acc_random_mean.max())
    min_y = np.minimum(unnot_greedy_mean.min(), unnot_random_mean.min())
    max_y = np.maximum(unnot_greedy_mean.max(), unnot_random_mean.max())
    x_eps = (max_x - min_x) * eps
    y_eps = (max_y - min_y) * eps
    axes.set_xlim(min_x-x_eps, max_x+x_eps)
    axes.set_ylim(min_y-y_eps, max_y+y_eps)

    random_area = -np.trapz(unnot_random_mean-unnot_random_mean[0], x=acc_random_mean)
    greedy_area = -np.trapz(unnot_greedy_mean-unnot_greedy_mean[0], x=acc_greedy_mean)

    rate = 1 - (greedy_area / random_area)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    plt.text(0.03, 0.03, f'Bypass Rate: {rate:.2f}',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform = axes.transAxes,
        color='black',
        fontsize=30,
        bbox=props)

    plt.savefig(f'q2_{graph_name}_{attacker}.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    ###############     Arguments     ###############
    parser = argparse.ArgumentParser()   
    parser.add_argument('--graph_name', type=str, default='cora', help='graph name: cora, citeeer,...')
    parser.add_argument('--attacker', type=str, default='random', help='attacker model name: random, dice, pgd, ...')
    parser.add_argument('--attack_rate', type=int, default=10, help='percentage of perturbing edges')
    args = parser.parse_args()

    q2_plot(graph_name=args.graph_name, attacker=args.attacker, attack_rate=args.attack_rate)

