#####################################
######## Paper Draft Cartoon ########
#####################################

#%%# Catalyzer

import os
import sys
path = os.path.dirname(os.path.realpath(__file__)) + '/../Resources'
sys.path.append(path)
import numpy as np
import torch
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
fig_resolution = 500 # {250, 375, 500, 750, 1000}
plt.rcParams['figure.dpi'] = fig_resolution
plt.rcParams['savefig.dpi'] = fig_resolution
fig_size_base = 3.75 # {2.5, 3.75, 5, 7.5, 10}

#%%# Data Preparation

from Fig0_Services import retrieve_mapes, retrieve_para_set_truths, retrieve_data_memo_auto_para_mem
from Fig0_Services import objective_fun_counter

data_paths = ['Shallow_Grid_1_N_Link'] # data_path
nooks = [0] # nook
acts = [7] # act
observers = [1] # observe
curbs = ['Mid'] # curb
mapes = retrieve_mapes(data_paths, acts, observers, curbs, verbose = True) # mape
para_key_sets = [ # para_key_set
    ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto']
]
para_value_sets = [ # para_value_set
    [(0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 250), (0, 1000), (300, 28800), (300, 28800), (30, 5*900), (30, 2*2100), (10*30, 43200), (30, 43200), (10*30, 43200), (30, 43200), (0, 1)]
]
para_set_modes = [0] # para_set_mode
para_set_truths = retrieve_para_set_truths(mapes, para_key_sets, para_value_sets, para_set_modes, verbose = True) # para_set_true

keys = ['data_path', 'nook', 'act', 'observe', 'curb', 'mape', 'para_set_true']
values = [data_paths, nooks, acts, observers, curbs, mapes, para_set_truths]
memos = {data_paths[_]: {key: value[_] for key, value in zip(keys, values)} for _ in range(len(data_paths))}

species = ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA'] # ['N', 'G', 'NP']
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.25
time_slit = 1
score_mini = 0
score_maxi = 1
score_slit = 0.05

cellulates = [(10, 10)] # (cell_layers, layer_cells) # [(1, 5), (2, 5), (3, 5), (5, 5), (7, 5), (5, 10), (5, 13), (5, 15), (5, 17), (10, 10), (10, 15), (15, 15), (20, 20)]
rules = [0, 1]
aim_NT = [1, 2/5]
aim_G = [1-_ for _ in aim_NT]
_auto_para_mem_set = [0, 3, 4, 7] # Decimals # 8 Instances
auto_para_mem_set_temp = [tuple(map(int, format(d, 'b'))) for d in _auto_para_mem_set] # Binaries
auto_para_mem_set = [tuple([0]*(3-len(tup)))+tup for tup in auto_para_mem_set_temp]
auto_para_mem_rules = [0 if _ == 0 else 1 for _ in _auto_para_mem_set]

#%%# Create Figure!

rows = 1
cols = 4
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)

lofts = ['A', 'P', 'M']
mutes = list()
for auto_para_mem in auto_para_mem_set:
    mute = ''
    for loft in lofts:
        loft_index = lofts.index(loft)
        mule = (auto_para_mem[loft_index] + 1) % 2
        mute += mule*loft
    mutes.append(mute)
_pats = ['ITWT', 'TM']
pats = [f"{_pats[int(mute != '')] + '-'*int(mute != '') + mute}" for mute in mutes]
# pats = ['TM XXX', 'TM XPM', 'TM AXX', 'ITWT'] # Oppress!

# Visualize Data Memo [Counter]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
meths = [('Auto', 'Para', 'Mem'), ('Autocrine', 'Paracrine', 'Membrane')] # {('Auto', 'Para', 'Mem'), ('Autocrine', 'Paracrine', 'Membrane')}

lot = 0 # {0, 1}
_meth = '_'.join(map(str, meths[lot]))
data_memo_auto_para_mem = retrieve_data_memo_auto_para_mem(memo, cellulates, auto_para_mem_set, meths[lot], verbose = True)

from copy import deepcopy
rule_index = 1 # rules.index(auto_para_mem_rules[auto_para_mem_index])
trajectory_set = data_memo_auto_para_mem[((10, 10), (1, 1, 1))] # Wild (Reference)
objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
# x = objective_fun.tau
w = deepcopy(counter)

synopses = {'Mean': 1, 'Sand': 1, 'Quantiles': 0}
quantiles = [0.5] # {[], [0.5], [0.25, 0.5, 0.75], [0.025, 0.5, 0.975]}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
cocos = ['turquoise', 'olive', 'navy', 'hotpink', 'maroon', 'blueviolet', 'coral'] # list(matplotlib.colors.CSS4_COLORS.keys())
coats = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:green', 'tab:cyan'] # list(matplotlib.colors.TABLEAU_COLORS.keys())
cocks = ['b', 'r'] # list(matplotlib.colors.BASE_COLORS.keys())
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
rows = 1
cols = 4
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
tits = ['[A]', '[B]', '[C]', '[D]']
bots = ['OFF', 'ON']
bot_cocos = ['crimson', 'forestgreen']
axe = axe_mat[0:1, 0:4]
for cellulate in cellulates:
    cellulate_index = cellulates.index(cellulate)
    cellulate_cells = cells[cellulate_index]
    for auto_para_mem in auto_para_mem_set[0:-1]:
        auto_para_mem_index = auto_para_mem_set.index(auto_para_mem)
        coco = cocos[auto_para_mem_index % len(cocos)]
        trajectory_set = data_memo_auto_para_mem[(cellulate, auto_para_mem)]
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
        x = objective_fun.tau
        y = counter
        row = auto_para_mem_index // cols
        col = auto_para_mem_index % cols
        _x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
        _y = np.linspace(0, 100, 11).astype('int')
        _axe = axe[row, col].inset_axes([0.5, 0.65, 0.5, 0.25])
        for clue in clues:
            clue_index = clues.index(clue)
            coat = coats[clue_index]
            if synopses['Mean']:
                if clue != 'U':
                    y_mean = np.mean(y[clue], 0).flatten()
                    w_mean = np.mean(w[clue], 0).flatten()
                else:
                    y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                    w_mean = np.mean(100-(w['NT']+w['G']), 0).flatten()
                axe[row, col].plot(x, w_mean, color = coat, linestyle = '--', marker = 'None', linewidth = 1, label = None, alpha = 0.75)
                axe[row, col].plot(x, y_mean, color = coat, linestyle = '-', marker = 'None', linewidth = 2, label = None, alpha = 0.75)
                if synopses['Sand']:
                    if clue != 'U':
                        y_sand = np.std(y[clue], 0).flatten()
                        w_sand = np.std(w[clue], 0).flatten()
                    else:
                        y_sand = np.std(100-(y['NT']+y['G']), 0).flatten()
                        w_sand = np.std(100-(w['NT']+w['G']), 0).flatten()
                    # y_net = y_mean-y_sand
                    # y_pot = y_mean+y_sand
                    # w_net = w_mean-w_sand
                    # w_pot = w_mean+w_sand
                    # axe[row, col].plot(x, w_net, color = coat, linestyle = '--', linewidth = 1, alpha = 0.25)
                    # axe[row, col].plot(x, w_pot, color = coat, linestyle = '--', linewidth = 1, alpha = 0.25)
                    # axe[row, col].plot(x, y_net, color = coat, linestyle = '--', linewidth = 1, alpha = 0.25)
                    # axe[row, col].plot(x, y_pot, color = coat, linestyle = '--', linewidth = 1, alpha = 0.25)
                    _axe.plot(x, w_sand, color = coat, linestyle = '--', marker = 'None', linewidth = 1, label = None, alpha = 0.75)
                    _axe.plot(x, y_sand, color = coat, linestyle = '-', marker = 'None', linewidth = 2, label = None, alpha = 0.75)
                elif synopses['Quantiles']:
                    pass
        _x_ticks = _x
        _y_ticks = range(0, 7)
        _axe.set_xticks(ticks = _x_ticks, labels = [None for _ in _x_ticks], fontsize = font_size_bet, rotation = 0)
        _axe.set_yticks(ticks = _y_ticks, labels = [_ if _ % 2 == 1 else None for _ in _y_ticks], fontsize = font_size_bet, rotation = 0)
        _axe.set_xlim(time_mini+24, time_maxi)
        _axe.set_ylim(min(_y_ticks), max(_y_ticks))
        _axe.set_title(label = r'$\sigma$', pad = 3, fontsize = font_size_bet)
        axe[row, col].set_title(label = pats[auto_para_mem_index] + ' ' + r'$\mu$', fontsize = font_size_bet)
        axe[row, col].set_title(label = tits[auto_para_mem_index], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
        _x_labels = [_ if _ % 8 == 0 else None for _ in _x] if row == 0 else [None for _ in _x]
        _y_labels = [_ if _ % 20 == 0 else None for _ in _y] if col == 0 else [None for _ in _y]
        axe[row, col].set_xticks(ticks = _x, labels = _x_labels, fontsize = font_size_bet, rotation = 0)
        axe[row, col].set_yticks(ticks = _y, labels = _y_labels, fontsize = font_size_bet, rotation = 0)
        if row == 0: axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
        if col == 0: axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
        axe[row, col].set_xlim(time_mini, time_maxi) # x_limes
        axe[row, col].set_ylim(0, 100) # y_limes
        axe[row, col].axhline(250, 0, 1, color = 'tab:gray', linestyle = '--', linewidth = 1, alpha = 1, label = _pats[0])
        axe[row, col].axhline(250, 0, 1, color = 'tab:gray', linestyle = '-', linewidth = 2, alpha = 1, label = _pats[1])
        axe[row, col].axhline(100*aim_NT[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[0], linestyle = ':', alpha = 0.25, label = None)
        axe[row, col].axhline(100*aim_G[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[1], linestyle = ':', alpha = 0.25, label = None)
        # axe[row, col].axhline(51.25, 4/(time_maxi-time_mini), 7/(time_maxi-time_mini), color = 'tab:gray', linestyle = '--', linewidth = 1, alpha = 1, label = None)
        # axe[row, col].axhline(48.75, 4/(time_maxi-time_mini), 7/(time_maxi-time_mini), color = 'tab:gray', linestyle = '-', linewidth = 2, alpha = 1, label = None)
        # axe[row, col].annotate(text = r'$\mu$', xy = (3, 50), xytext = (3, 50), color = 'black', fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'right', verticalalignment = 'center')
        if row == 0 and col != 0:
            axe[row, col].annotate(text = f'Auto {bots[auto_para_mem[0]]}', xy = (0.5, 55), xytext = (0.5, 55), color = bot_cocos[auto_para_mem[0]], fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
            axe[row, col].annotate(text = f'Para {bots[auto_para_mem[1]]}', xy = (0.5, 50), xytext = (0.5, 50), color = bot_cocos[auto_para_mem[1]], fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
            axe[row, col].annotate(text = f'Mem {bots[auto_para_mem[2]]}', xy = (0.5, 45), xytext = (0.5, 45), color = bot_cocos[auto_para_mem[2]], fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
        else:
            axe[row, col].annotate(text = f'Auto {bots[auto_para_mem[0]]}', xy = (37, 25), xytext = (37, 25), color = bot_cocos[auto_para_mem[0]], fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
            axe[row, col].annotate(text = f'Para {bots[auto_para_mem[1]]}', xy = (37, 20), xytext = (37, 20), color = bot_cocos[auto_para_mem[1]], fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
            axe[row, col].annotate(text = f'Mem {bots[auto_para_mem[2]]}', xy = (37, 15), xytext = (37, 15), color = bot_cocos[auto_para_mem[2]], fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
        # axe[row, col].grid(alpha = 0.125)
        axe[row, col].legend(fontsize = font_size_chi, frameon = True, framealpha = 0.25, loc = (0.075, 0.875))
# plt.show()

# Visualize Data Memo [Raw Counter]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
meths = [('Auto', 'Para', 'Mem'), ('Autocrine', 'Paracrine', 'Membrane')] # {('Auto', 'Para', 'Mem'), ('Autocrine', 'Paracrine', 'Membrane')}

# lot = 0 # {0, 1}
_meth = '_'.join(map(str, meths[lot]))
data_memo_auto_para_mem = retrieve_data_memo_auto_para_mem(memo, cellulates, auto_para_mem_set, meths[lot], verbose = True)

rule_index = 1 # rules.index(auto_para_mem_rules[auto_para_mem_index])

synopses = {'Mean': 1, 'Sand': 1, 'Quantiles': 0}
quantiles = [0.5] # {[], [0.5], [0.25, 0.5, 0.75], [0.025, 0.5, 0.975]}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
cocos = ['turquoise', 'olive', 'navy', 'hotpink', 'maroon', 'blueviolet', 'coral'] # list(matplotlib.colors.CSS4_COLORS.keys())
coats = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:green', 'tab:cyan'] # list(matplotlib.colors.TABLEAU_COLORS.keys())
bears = ['tab:blue', 'tab:orange', 'olive']
cocks = ['b', 'r'] # list(matplotlib.colors.BASE_COLORS.keys())
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
rows = 1
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
# tits = ['[A]', '[B]', '[C]', '[D]']
epoch = 48*int(1/time_delta)
alpha = 1-1/4-1/8
axe = axe_mat[0:1, 3:4]
row = 0
col = 0
for cellulate in cellulates:
    cellulate_index = cellulates.index(cellulate)
    cellulate_cells = cells[cellulate_index]
    for auto_para_mem in auto_para_mem_set:
        auto_para_mem_index = auto_para_mem_set.index(auto_para_mem)
        coco = cocos[auto_para_mem_index % len(cocos)]
        trajectory_set = data_memo_auto_para_mem[(cellulate, auto_para_mem)]
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
        t = objective_fun.tau
        x = auto_para_mem_index
        y = counter
        if synopses['Mean'] and synopses['Sand']:
            y_mean_NT = np.mean(y['NT'], 0).flatten()
            y_sand_NT = np.std(y['NT'], 0).flatten()
            y_mean_G = np.mean(y['G'], 0).flatten()
            y_sand_G = np.std(y['G'], 0).flatten()
            y_mean_U = np.mean(100-(y['NT']+y['G']), 0).flatten()
            y_sand_U = np.std(100-(y['NT']+y['G']), 0).flatten()
            axe[row, col].set_title(label = 'Time = 48 Hours', fontsize = font_size_bet)
            axe[row, col].set_title(label = tits[-1], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
            axe[row, col].bar(x, y_mean_NT[epoch], 0.75, y_mean_G[epoch], color = coats[0], alpha = alpha, label = clue_labels[0] if auto_para_mem_index == 0 else None)
            axe[row, col].bar(x, y_mean_G[epoch], 0.75, 0, color = coats[1], alpha = alpha, label = clue_labels[1] if auto_para_mem_index == 0 else None)
            axe[row, col].bar(x, y_mean_U[epoch], 0.75, y_mean_NT[epoch]+y_mean_G[epoch], color = coats[2], alpha = alpha, label = clue_labels[2] if auto_para_mem_index == 0 else None)
            alp = axe[row, col].errorbar(x, y_mean_G[epoch], yerr = y_sand_G[epoch], fmt = 'None', ecolor = bears[1], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            bet = axe[row, col].errorbar(x, y_mean_G[epoch]+y_mean_NT[epoch], yerr = y_sand_NT[epoch], fmt = 'None', ecolor = bears[0], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            chi = axe[row, col].errorbar(x, y_mean_G[epoch]+y_mean_NT[epoch]+y_mean_U[epoch], yerr = y_sand_U[epoch], fmt = 'None', ecolor = bears[2], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            alp[1][0].set_marker('_')
            bet[1][0].set_marker('_')
            chi[1][0].set_marker('_')
            alp[1][0].set_markersize(font_size_chi)
            bet[1][0].set_markersize(font_size_chi)
            chi[1][0].set_markersize(font_size_chi)
        elif synopses['Quantiles']:
            pass
_x = range(0, len(auto_para_mem_set))
_y = np.linspace(0, 100, 11).astype('int')
axe[row, col].set_xticks(ticks = _x, labels = pats, fontsize = font_size_bet, rotation = 0) # {0, 15, 30, 45, 60, 75, 90}
axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 20 == 0 else None for _ in _y], fontsize = font_size_bet)
# axe[row, col].set_xlabel('TM = Theoretical Mutant\nITWT = Inferential-Theoretical Wild Type', fontsize = font_size_bet)
axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
axe[row, col].set_xlim(-0.625, len(_x)-0.375) # x_limes
axe[row, col].set_ylim(0, 100) # y_limes
axe[row, col].legend(loc = 'center left', fontsize = font_size_chi+2)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig6'
fig_forts = ['tiff', 'svg', 'eps', 'pdf'] # {TIFF, SVG, EPS, PDF}
fig_keros = [{'compression': 'tiff_lzw'}, None, None, None]
fig_flags = [fig_kero is not None for fig_kero in fig_keros]

for fig_fort in fig_forts:
    fig_fort_index = fig_forts.index(fig_fort)
    fig_kero = fig_keros[fig_fort_index]
    fig_flag = fig_flags[fig_fort_index]
    fig_clue = f'{fig_path}{fig_nick}.{fig_fort}'
    print(fig_clue)
    if fig_flag:
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort, pil_kwargs = fig_kero)
    else:
        fig.savefig(fname = fig_clue, dpi = fig_resolution, format = fig_fort)
