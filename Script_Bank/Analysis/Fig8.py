##############################
######## Shallow Grid ########
##############################

#%%# Catalyzer

import os
import sys
path = os.path.dirname(os.path.realpath(__file__)) + '/../Resources'
sys.path.append(path)
path = os.path.dirname(os.path.realpath(__file__))
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

from Fig0_Services import retrieve_mapes, retrieve_para_set_truths
from Fig0_Services import objective_fun_counter

data_paths = ['Shallow_Grid_1_N_Link'] # data_path
nooks = [0, 0] # nook
acts = [7, 7] # act
observers = [1, 1] # observe
curbs = ['Mid', 'Mid'] # curb
mapes = retrieve_mapes(data_paths, acts, observers, curbs, verbose = True) # mape
para_key_sets = [ # para_key_set
    ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto'],
    ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'N_MRNA', 'G_MRNA', 'FC_MRNA', 'N', 'G', 'FC', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto']
]
para_value_sets = [ # para_value_set
    [(0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 250), (0, 1000), (300, 28800), (300, 28800), (30, 5*900), (30, 2*2100), (10*30, 43200), (30, 43200), (10*30, 43200), (30, 43200), (0, 1)],
    [(0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 250), (0, 250), (0, 200), (0, 1000), (0, 1000), (0, 800), (300, 28800), (300, 28800), (30, 5*900), (30, 2*2100), (10*30, 43200), (30, 43200), (10*30, 43200), (30, 43200), (0, 1)]
]
para_set_modes = [0, 0] # para_set_mode
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

_uniform = [0, 100, 200] # [0, 1, 5, 10, 25, 50, 75, 100, 150, 200]

_initiate_set_keys = ['MRNA', 'PRO']
_initiate_set_values = [
    [[100]],
    [[100]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

#%%# Retrieve Data [Initiate]

def retrieve_data_cellulate_initiate(memo, cellulate, initiate, reiteration, verbose = False):
    from Utilities import simul_data_load
    if verbose: print(f"{' '*8}{cellulate}{' '*8}{initiate}")
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
    tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate_initiate = (theta_set, trajectory_set)
    return data_cellulate_initiate

def retrieve_data_memo_initiate(memo, cellulates, initiate_set, uniform, verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    keys = list(initiate_set.keys())
    values = list(initiate_set.values())
    initiate_set_temp = list(zip(*values))
    data_memo_initiate = {(cellulate, initiate_temp): None for cellulate in cellulates for initiate_temp in initiate_set_temp}
    for cellulate in cellulates:
        for initiate_temp in initiate_set_temp:
            initiate = {keys[index]: initiate_temp[index] for index in range(len(keys))}
            _initiate = '_'.join(map(str, [f'{key}_{value}' for key, value in initiate.items()]))
            reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}_Initiate_{_initiate}_Uniform_{uniform}'
            data_cellulate_initiate = retrieve_data_cellulate_initiate(memo, cellulate, initiate, reiteration, verbose)
            data_memo_initiate[(cellulate, initiate_temp)] = data_cellulate_initiate[1]
    return data_memo_initiate

#%%# Create Figure!

rows = 2
cols = 2
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = True)
# axe_mat[0, 2].remove()

# Visualize Data Memo [Counter Mean]

_uniform = [0, 100, 200] # [0, 1, 5, 10, 25, 50, 75, 100, 150, 200]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

synopses = {'Mean': 1, 'Sand': 0}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
# cocos = ['tab:gray', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:olive', 'tab:cyan']
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (5, 3)), (0, (1, 1, 1, 1))] # ['-', (0, (5, 3)), (0, (3, 2, 1, 2))]
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
rows = 1
cols = 1 # len(clues) # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = True, sharey = True, squeeze = False, figsize = fig_size, layout = 'constrained')
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[1:2, 0:1]
for uniform in _uniform:
    uniform_index = _uniform.index(uniform)
    row = 0
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, uniform, verbose = True)
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        initiate_set_keys = list(initiate_set_temp.keys())
        initiate_set_values = list(initiate_set_temp.values())
        initiate_set = list(zip(*initiate_set_values))
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            trajectory_set = data_memo_initiate[(cellulate, initiate)]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            y = counter if not normalize else {key: counter[key]/normalization for key in counter.keys()}
            coco = cocos[uniform_index % len(cocos)]
            cap = caps[initiate_index // len(cocos)]
            edge = edges[uniform_index]
            row = 0
            col = 0
            for clue in clues:
                clue_index = clues.index(clue)
                label = f'{uniform}%' if clue_index == 0 else None
                if synopses['Mean']:
                    if clue != 'U':
                        coco = cocos[clue_index]
                        y_mean = np.mean(y[clue], 0).flatten()
                    else:
                        coco = cocos[8]
                        y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                    axe[row, col].plot(x, y_mean, color = 'k', fillstyle = cap, linestyle = edge, marker = 'None', linewidth = 1, label = label, alpha = 0.5)
                    axe[row, col].plot(x, y_mean, color = coco, fillstyle = cap, linestyle = edge, marker = 'None', linewidth = 1, label = None)
                    if synopses['Sand']:
                        if clue != 'U':
                            y_sand = np.std(y[clue], 0).flatten()
                        else:
                            y_sand = np.std(100-(y['NT']+y['G']), 0).flatten()
                        y_net = y_mean-y_sand
                        y_pot = y_mean+y_sand
                        axe[row, col].plot(x, y_net, color = coco, linestyle = '--', linewidth = 1, alpha = 0.5)
                        axe[row, col].plot(x, y_pot, color = coco, linestyle = '--', linewidth = 1, alpha = 0.5)
                _x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
                _y = np.linspace(0, 100, 11).astype('int')
                axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 20 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                if row == rows-1: axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
                if col == 0: axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
                axe[row, col].set_xlim(time_mini, time_maxi) # x_limes
                axe[row, col].set_ylim(0, 100) # y_limes
                # axe[row, col].grid(alpha = 0.125)
    axe[row, col].axhline(aim_NT[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'b' if not normalize else cocks[0], linestyle = (0, (3, 2, 1, 2)), linewidth = 1, alpha = 0.125, label = 'EPI Target' if uniform_index == len(_uniform) else None, zorder = 10)
    axe[row, col].axhline(aim_G[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'r' if not normalize else cocks[1], linestyle = (0, (3, 2, 1, 2)), linewidth = 1, alpha = 0.125, label = 'PRE Target' if uniform_index == len(_uniform) else None, zorder = 10)
    axe[row, col].legend(fontsize = font_size_chi)
    axe[row, col].set_title(label = 'μ', fontsize = font_size_bet)
    axe[row, col].set_title(label = '[C]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
# plt.show()

# Visualize Data Memo [Counter Sand]

_uniform = [0, 100, 200] # [0, 1, 5, 10, 25, 50, 75, 100, 150, 200]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

synopses = {'Mean': 0, 'Sand': 1}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
# cocos = ['tab:gray', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:olive', 'tab:cyan']
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (5, 3)), (0, (1, 1, 1, 1))] # ['-', (0, (5, 3)), (0, (3, 2, 1, 2))]
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
rows = 1
cols = 1 # len(clues) # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = True, sharey = True, squeeze = False, figsize = fig_size, layout = 'constrained')
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[1:2, 1:2]
_axe = axe[0, 0].inset_axes([0.5, 0, 0.5, 0.2])
for uniform in _uniform:
    uniform_index = _uniform.index(uniform)
    row = 0
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, uniform, verbose = True)
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        initiate_set_keys = list(initiate_set_temp.keys())
        initiate_set_values = list(initiate_set_temp.values())
        initiate_set = list(zip(*initiate_set_values))
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            trajectory_set = data_memo_initiate[(cellulate, initiate)]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            y = counter if not normalize else {key: counter[key]/normalization for key in counter.keys()}
            coco = cocos[uniform_index % len(cocos)]
            cap = caps[initiate_index // len(cocos)]
            edge = edges[uniform_index]
            row = 0
            col = 0
            for clue in clues:
                clue_index = clues.index(clue)
                label = f'{uniform}%' if clue_index == 0 else None
                if synopses['Sand']:
                    if clue != 'U':
                        coco = cocos[clue_index]
                        y_mean = np.mean(y[clue], 0).flatten()
                        y_sand = np.std(y[clue], 0).flatten()
                    else:
                        coco = cocos[8]
                        y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                        y_sand = np.std(100-(y['NT']+y['G']), 0).flatten()
                    axe[row, col].plot(x, y_sand, color = 'k', fillstyle = cap, linestyle = edge, marker = 'None', linewidth = 1, label = label, alpha = 0.5)
                    axe[row, col].plot(x, y_sand, color = coco, linestyle = edge, linewidth = 1, label = None)
                    _axe.plot(x, y_sand/y_mean, color = coco, linestyle = edge, linewidth = 1, label = None)
                _x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
                _y = np.round(np.linspace(0, 5, 11), 2) # np.linspace(0, 5, 11).astype('int')
                axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_yticks(ticks = _y, labels = [int(_) if _ % 1 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                _y = np.round(np.linspace(0, 0.5, 5),3)
                _axe.set_xticks(ticks = [], labels = [], fontsize = font_size_chi)
                _axe.set_yticks(ticks = _y, labels = [_ if _ > 0 else None for _ in _y], fontsize = font_size_chi)
                if row == rows-1: axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
                if col == 0: axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
                axe[row, col].set_xlim(time_mini, time_maxi) # x_limes
                axe[row, col].set_ylim(0, 5) # y_limes
                _axe.set_xlim(time_mini+24, time_maxi) # x_limes
                _axe.set_ylim(0, 0.5) # y_limes
                _axe.set_ylabel('Ratio', fontsize = font_size_bet)
                # axe[row, col].grid(alpha = 0.125)
    # axe[row, col].axhline(aim_NT[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'b' if not normalize else cocks[0], linestyle = (0, (3, 2, 1, 2)), linewidth = 1, alpha = 0.125, label = 'N Target' if uniform_index == len(_uniform) else None, zorder = 10)
    # axe[row, col].axhline(aim_G[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'r' if not normalize else cocks[1], linestyle = (0, (3, 2, 1, 2)), linewidth = 1, alpha = 0.125, label = 'G Target' if uniform_index == len(_uniform) else None, zorder = 10)
    axe[row, col].legend(loc = (0.75, 0.575), fontsize = font_size_chi)
    axe[row, col].set_title(label = 'σ', fontsize = font_size_bet)
    axe[row, col].set_title(label = '[D]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    _axe.set_title(label = 'σ/μ', fontsize = font_size_bet)
# plt.show()

# Visualize Data Memo [Initial Condition]

_uniform = [0, 100, 200][::-1] # [0, 1, 5, 10, 25, 50, 75, 100, 150, 200]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

synopses = {'Mean': 1, 'Sand': 0}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
# cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
# cocks = list(matplotlib.colors.BASE_COLORS.keys())
cocos = ['limegreen', 'tab:purple', 'tab:pink'] # ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# cocks = ['c', 'g', 'm', 'b', 'r', 'y', 'k', 'w']
marks = ['s', 'D', 'o']
_species_selection = [['N', 'G']] # [['N', 'G'], ['N_MRNA', 'G_MRNA']]
_species_selection_labels = [['NANOG', 'GATA6']] # [['N Protein', 'G Protein'], ['N MRNA', 'G MRNA']]
_species_selection_limes = [(0, 1000), (0, 250)]
rows = 1
cols = 2 # len(clues) # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
mark_size = 11
axe = axe_mat[0:1, 0:2]
_axe_x = np.empty_like(axe)
_axe_x[0, 0] = axe[0, 0].inset_axes([0, 1, 1, 0.2])
# _axe_x[0, 1] = axe[0, 1].inset_axes([0, 1, 1, 0.25])
_axe_y = np.empty_like(axe)
_axe_y[0, 0] = axe[0, 0].inset_axes([1, 0, 0.2, 1])
# _axe_y[0, 1] = axe[0, 1].inset_axes([1, 0, 0.25, 1])
for uniform in _uniform:
    uniform_index = _uniform.index(uniform)
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, uniform, verbose = True)
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        initiate_set_keys = list(initiate_set_temp.keys())
        initiate_set_values = list(initiate_set_temp.values())
        initiate_set = list(zip(*initiate_set_values))
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            trajectory_set = data_memo_initiate[(cellulate, initiate)]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            objective_fun.apply(tau_mini = time_mini, tau_maxi = time_maxi)
            t = objective_fun.tau
            z = objective_fun.data
            coco = cocos[uniform_index % len(cocos)]
            mark = marks[uniform_index]
            for species_selection in _species_selection:
                species_selection_index = _species_selection.index(species_selection)
                species_selection_labels = _species_selection_labels[species_selection_index]
                species_selection_limes = _species_selection_limes[species_selection_index]
                row = 0
                col = species_selection_index
                # _axe_x = axe[row, col].inset_axes([0, 1, 1, 0.25])
                # _axe_y = axe[row, col].inset_axes([1, 0, 0.25, 1])
                x_index = species.index(species_selection[0])
                y_index = species.index(species_selection[1])
                x = z[:, x_index, 0, :].flatten()
                y = z[:, y_index, 0, :].flatten()
                axe[row, col].fill_between([0, 1000], 0, 1000, color = 'tab:olive', alpha = 0.025)
                axe[row, col].scatter(x, y, s = mark_size, c = coco, marker = mark, edgecolors = 'none', alpha = 0.75, label = f'{uniform}%')
                hoax = [200+5*np.sqrt(200), 1000-5*np.sqrt(1000)]
                axe[row, col].hlines(hoax, [500-5*np.sqrt(500), 0], [1000, 200+5*np.sqrt(200)], colors = 'tab:olive', linestyles = '-', linewidths = 1, label = None)
                axe[row, col].hlines(hoax, [500-5*np.sqrt(500), 0], [1000, 200+5*np.sqrt(200)], colors = ['tab:blue', 'tab:orange'], linestyles = '--', linewidths = 1, label = None)
                axe[row, col].fill_between([0, 200+5*np.sqrt(200)], 1000-5*np.sqrt(1000), 1000, color = 'tab:orange', alpha = 0.1)
                very = [200+5*np.sqrt(200), 500-5*np.sqrt(500)]
                axe[row, col].vlines(very, [1000-5*np.sqrt(1000), 0], [1000, 200+5*np.sqrt(200)], colors = 'tab:olive', linestyles = '-', linewidths = 1, label = None)
                axe[row, col].vlines(very, [1000-5*np.sqrt(1000), 0], [1000, 200+5*np.sqrt(200)], colors = ['tab:orange', 'tab:blue'], linestyles = '--', linewidths = 1, label = None)
                axe[row, col].fill_between([500-5*np.sqrt(500), 1000], 0, 200+5*np.sqrt(200), color = 'tab:blue', alpha = 0.1)
                _x = np.linspace(species_selection_limes[0], species_selection_limes[1], 11).astype('int')
                _y = np.linspace(species_selection_limes[0], species_selection_limes[1], 11).astype('int')
                axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 200 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 200 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_xlabel(species_selection_labels[0], fontsize = font_size_bet)
                axe[row, col].set_ylabel(species_selection_labels[1], fontsize = font_size_bet)
                axe[row, col].set_xlim(species_selection_limes[0], species_selection_limes[1]) # x_limes
                axe[row, col].set_ylim(species_selection_limes[0], species_selection_limes[1]) # y_limes
                weights = None if uniform != 0 else 0.25*np.ones(tuple([len(x)]))
                _axe_x[row, col].hist(x, bins = species_selection_limes[1]-species_selection_limes[0], range = (species_selection_limes[0], species_selection_limes[1]), density = False, weights = weights, histtype = 'stepfilled', color = coco, alpha = 0.75)
                _axe_y[row, col].hist(y, bins = species_selection_limes[1]-species_selection_limes[0], range = (species_selection_limes[0], species_selection_limes[1]), density = False, weights = weights, histtype = 'stepfilled', orientation = 'horizontal', color = coco, alpha = 0.75)
                _x = np.linspace(species_selection_limes[0], species_selection_limes[1], 11).astype('int')
                _y = np.round(np.linspace(0, 1, 3), 2)
                _axe_x[row, col].set_xticks(ticks = _x, labels = [], fontsize = font_size_bet, rotation = 0)
                _axe_x[row, col].set_yticks(ticks = [], labels = [], fontsize = font_size_bet, rotation = 0) # _axe_x[row, col].set_yticks(ticks = _y, labels = [_ if _ > 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                _axe_x[row, col].set_xlim(species_selection_limes[0], species_selection_limes[1])
                _x = np.round(np.linspace(0, 1, 3), 2)
                _y = np.linspace(species_selection_limes[0], species_selection_limes[1], 11).astype('int')
                _axe_y[row, col].set_xticks(ticks = [], labels = [], fontsize = font_size_bet, rotation = 0) # _axe_y[row, col].set_xticks(ticks = _x, labels = [_ if _ > 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                _axe_y[row, col].set_yticks(ticks = _y, labels = [], fontsize = font_size_bet, rotation = 0)
                _axe_y[row, col].set_ylim(species_selection_limes[0], species_selection_limes[1])
                # axe[row, col].grid(alpha = 0.125)
                axe[row, col].set_title(label = 'Initial Condition Distribution (ICD)', loc = 'center', fontsize = font_size_bet)
                axe[row, col].set_title(label = '[A]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold', position = (-0.1875, 1))
                axe[row, col].legend(bbox_to_anchor = (1, 1), loc = 'lower left', fontsize = font_size_chi)
                axe[row, col].text(x = (1000+500-5*np.sqrt(500))/2, y = (200+5*np.sqrt(200))/2, s = 'EPI', alpha = 1, color = 'tab:blue', fontsize = font_size_bet, horizontalalignment = 'center', verticalalignment = 'center')
                axe[row, col].text(x = (200+5*np.sqrt(200))/2, y = (1000+1000-5*np.sqrt(1000))/2, s = 'PRE', alpha = 1, color = 'tab:orange', fontsize = font_size_bet, horizontalalignment = 'center', verticalalignment = 'center')
                axe[row, col].text(x = 1000*(1-1/8), y = 1000*(1-1/8), s = 'UND', alpha = 1, color = 'tab:olive', fontsize = font_size_bet, horizontalalignment = 'center', rotation = 45, rotation_mode = 'anchor', verticalalignment = 'center')
    # axe[row, col].set_title(label = '', fontsize = font_size_bet)
# plt.show()

# Visualize Data Memo [Raw Counter] [Nook]

_uniform = [0, 1, 5, 10, 25, 50, 75, 100, 150, 200] # [0, 1, 5, 10, 25, 50, 75, 100, 150, 200]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

i = 0
_memo = data_paths[i]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

synopses = {'Mean': 1, 'Sand': 0, 'Quantiles': 0}
quantiles = [0.5] # {[], [0.5], [0.25, 0.5, 0.75], [0.025, 0.5, 0.975]}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
# cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
cocos = ['tab:blue', 'tab:orange', 'tab:olive'] # ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
bears = ['tab:blue', 'tab:orange', 'olive']
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (3, 7))]
clues = ['NT', 'G']
rows = 1
cols = 1 # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
epoch = 48*int(1/time_delta)
alpha = 0.75-0.25/2
axe = axe_mat[0:1, 1:2]
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
for uniform in _uniform:
    uniform_index = _uniform.index(uniform)
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, uniform, verbose = True)
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        initiate_set_keys = list(initiate_set_temp.keys())
        initiate_set_values = list(initiate_set_temp.values())
        initiate_set = list(zip(*initiate_set_values))
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            trajectory_set = data_memo_initiate[(cellulate, initiate)]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            t = objective_fun.tau
            x = uniform_index
            y = counter if not normalize else {key: counter[key]/normalization for key in counter.keys()}
            coco = cocos[initiate_index % len(cocos)]
            cap = caps[initiate_index // len(cocos)]
            edge = edges[initiate_index // len(cocos)]
            for row in range(rows):
                col = 0
                y_mean_NT = np.mean(y['NT'], 0).flatten()
                y_sand_NT = np.std(y['NT'], 0).flatten()
                y_mean_G = np.mean(y['G'], 0).flatten()
                y_sand_G = np.std(y['G'], 0).flatten()
                y_mean_D = np.mean(100-(y['NT']+y['G']), 0).flatten()
                y_sand_D = np.std(100-(y['NT']+y['G']), 0).flatten()
                axe[row, col].set_title(label = 'Time = 48 Hours', fontsize = font_size_bet)
                axe[row, col].set_title(label = '[B]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
                axe[row, col].bar(x, y_mean_NT[epoch], 0.75, y_mean_G[epoch], color = cocos[0], alpha = alpha, label = 'EPI' if uniform_index == 0 else None)
                axe[row, col].bar(x, y_mean_G[epoch], 0.75, 0, color = cocos[1], alpha = alpha, label = 'PRE' if uniform_index == 0 else None)
                axe[row, col].bar(x, y_mean_D[epoch], 0.75, y_mean_NT[epoch]+y_mean_G[epoch], color = cocos[2], alpha = alpha, label = 'UND' if uniform_index == 0 else None)
                alp = axe[row, col].errorbar(x, y_mean_G[epoch], yerr = y_sand_G[epoch], fmt = 'None', ecolor = bears[1], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
                bet = axe[row, col].errorbar(x, y_mean_G[epoch]+y_mean_NT[epoch], yerr = y_sand_NT[epoch], fmt = 'None', ecolor = bears[0], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
                chi = axe[row, col].errorbar(x, y_mean_G[epoch]+y_mean_NT[epoch]+y_mean_D[epoch], yerr = y_sand_D[epoch], fmt = 'None', ecolor = bears[2], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
                alp[1][0].set_marker('_')
                bet[1][0].set_marker('_')
                chi[1][0].set_marker('_')
                alp[1][0].set_markersize(7)
                bet[1][0].set_markersize(7)
                chi[1][0].set_markersize(7)
                _x = np.arange(0, len(_uniform)).astype('int')
                _y = np.linspace(50, 100, 11).astype('int')
                # tag = [int(_*mapes[i][13]/100) for _ in _initiate_set[i]['FC']] if percentage[i] else _initiate_set[i]['FC']
                tag = None
axe[row, col].set_xticks(ticks = _x, labels = [f'{_}%' for _ in _uniform], fontsize = font_size_bet, rotation = 45)
axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 10 == 0 else None for _ in _y], fontsize = font_size_bet)
axe[row, col].set_xlabel('Initial Condition Perturbation (ICP)', fontsize = font_size_bet)
axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
axe[row, col].set_xlim(-0.625, len(_uniform)-0.375) # x_limes
axe[row, col].set_ylim(50, 100) # y_limes
axe[row, col].legend(loc = 'center right', fontsize = font_size_chi)
# axe[row, col].axhline(aim_NT[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'b' if not normalize else cocks[0], linestyle = '--', alpha = 0.025)
# axe[row, col].axhline(aim_G[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'r' if not normalize else cocks[0], linestyle = '--', alpha = 0.025)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig8'
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
