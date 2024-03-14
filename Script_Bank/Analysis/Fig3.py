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

from Fig0_Services import retrieve_mapes, retrieve_para_set_truths, retrieve_data_memo_initiate
from Fig0_Services import objective_fun_cellular_stats, objective_fun_classifier, objective_fun_counter

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
time_slit = 0
score_mini = 0
score_maxi = 1
score_slit = 0.05

cellulates = [(10, 10)] # (cell_layers, layer_cells) # [(1, 5), (2, 5), (3, 5), (5, 5), (7, 5), (5, 10), (5, 13), (5, 15), (5, 17), (10, 10), (10, 15), (15, 15), (20, 20)]
rules = [0, 1]
aim_NT = [1, 2/5]
aim_G = [1-_ for _ in aim_NT]

_initiate_set_keys = ['FC_MRNA', 'FC'] # ['MRNA', 'PRO']
_initiate_set_values = [ # [0, 25, 50, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 150, 175, 200, 225, 250]
    [[0]*14], # FC_MRNA
    [[0, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]] # FC
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

# Create Figure!

rows = 2
cols = 2
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = True)

# Visualize Data Memo [Cellular Stats Whole]

_initiate_set_keys = ['FC_MRNA', 'FC'] # ['MRNA', 'PRO']
_initiate_set_values = [
    [[0], [100]],
    [[0], [100]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

synopses = {'Mean': 1, 'Sand': 1}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['none', 'none']
edges = ['-', '--']
species_selection = ['NT', 'G', 'FT']
species_labels = ['NANOG', 'GATA6', 'FGF4']
_wait = [0] # [0, 4, 8, 12, 16, 24, 32, 40]
rows = 1 # len(_wait)
cols = 1 # len(species_selection)
# fig_size = (cols*fig_size_base, rows*fig_size_base)
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[1, 0]
for wait in _wait:
    row = 0 # _wait.index(wait)
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, wait, verbose = True)
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
            cellular_stats = objective_fun_cellular_stats(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            classifier = objective_fun_classifier(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            coco = cocos[initiate_index % len(cocos)]
            cap = caps[initiate_index // len(cocos)]
            edge = edges[initiate_index // len(cocos)]
            for spa in species_selection:
                spa_index = species_selection.index(spa)
                col = 0 # species_selection.index(spa)
                y_cellular_stats = cellular_stats[spa]
                if synopses['Mean']:
                    y_mean = np.mean(np.mean(y_cellular_stats, 0), 1)
                    axe.plot(x, y_mean, color = cocos[spa_index], fillstyle = cap, linestyle = edges[0], marker = 'None', linewidth = 2, label = species_labels[spa_index])
                if synopses['Sand']:
                    y_mean = np.mean(np.mean(y_cellular_stats, 0), 1)
                    y_sand = np.array([np.std(y_cellular_stats[:, _, :]) for _ in range(y_cellular_stats.shape[1])])
                    axe.plot(x, y_mean-y_sand, color = cocos[spa_index], fillstyle = cap, linestyle = edges[1], marker = 'None', linewidth = 1, alpha = 0.5)
                    axe.plot(x, y_mean+y_sand, color = cocos[spa_index], fillstyle = cap, linestyle = edges[1], marker = 'None', linewidth = 1, alpha = 0.5)
                _x = np.array([_ for _ in x if _ % 4 == 0], 'int')
                _y = np.linspace(0, 1000, 11).astype('int')
                # if row == 0: axe[row, col].set_title(label = spa)
                axe.set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                axe.set_yticks(ticks = _y, labels = [_ if _ % 200 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                if row == rows-1: axe.set_xlabel('Time [Hour]', fontsize = font_size_bet)
                if col == 0: axe.set_ylabel('Molecule Count', fontsize = font_size_bet)
                axe.set_xlim(time_mini-time_slit, time_maxi+time_slit) # x_limes
                axe.set_ylim(0, 1000) # y_limes
                # axe[row, col].grid(alpha = 0.125)
                axe.set_title(label = 'Cell Level μ ± σ', fontsize = font_size_bet)
                axe.set_title(label = '[C]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
                axe.legend(fontsize = font_size_chi, loc = 'upper left', title = 'Total', title_fontsize = font_size_chi)
# fig.suptitle(t = 'Cellular Stats')
# plt.show()

# Visualize Data Memo [Counter]

_initiate_set_keys = ['FC_MRNA', 'FC'] # ['MRNA', 'PRO']
_initiate_set_values = [
    [[0], [100]],
    [[0], [100]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]
data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, 0, verbose = True)

synopses = {'Mean': 1, 'Sand': 1}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (3, 7))]
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
rows = 1
cols = 1 # len(clues) # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[1, 1]
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
        coco = cocos[initiate_index % len(cocos)]
        cap = caps[initiate_index // len(cocos)]
        edge = edges[initiate_index // len(cocos)]
        row = 0
        col = 0
        for clue in clues:
            clue_index = clues.index(clue)
            if synopses['Mean']:
                if clue != 'U':
                    coco = cocos[clue_index]
                    y_mean = np.mean(y[clue], 0).flatten()
                else:
                    coco = cocos[8]
                    y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                axe.plot(x, y_mean, color = coco, fillstyle = cap, linestyle = edges[0], marker = 'None', linewidth = 2, label = clue_labels[clue_index])
                if synopses['Sand']:
                    if clue != 'U':
                        y_sand = np.std(y[clue], 0).flatten()
                    else:
                        y_sand = np.std(100-(y['NT']+y['G']), 0).flatten()
                    y_net = y_mean-y_sand
                    y_pot = y_mean+y_sand
                    axe.plot(x, y_net, color = coco, linestyle = '--', linewidth = 1, alpha = 0.5)
                    axe.plot(x, y_pot, color = coco, linestyle = '--', linewidth = 1, alpha = 0.5)
            _x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
            _y = np.linspace(0, 100, 11).astype('int')
            axe.set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
            axe.set_yticks(ticks = _y, labels = [_ if _ % 20 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
            if row == rows-1: axe.set_xlabel('Time [Hour]', fontsize = font_size_bet)
            if col == 0: axe.set_ylabel('Cell Count', fontsize = font_size_bet)
            axe.set_xlim(time_mini-time_slit, time_maxi+time_slit) # x_limes
            axe.set_ylim(0, 100) # y_limes
            # axe[row, col].grid(alpha = 0.125)
        axe.axhline(aim_NT[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'b' if not normalize else cocks[0], linestyle = ':', linewidth = 2, alpha = 0.25, label = 'EPI Target')
        axe.axhline(aim_G[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'r' if not normalize else cocks[1], linestyle = ':', linewidth = 2, alpha = 0.25, label = 'PRE Target')
        axe.set_title(label = 'Tissue Level μ ± σ', fontsize = font_size_bet)
        axe.set_title(label = '[D]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
        axe.legend(fontsize = font_size_chi, loc = 'upper right')

# Visualize Data Memo [Cellular Stats]

_initiate_set_keys = ['FC_MRNA', 'FC'] # ['MRNA', 'PRO']
_initiate_set_values = [
    [[0], [100]],
    [[0], [100]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

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
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['none', 'none']
edges = ['-', '--']
species_selection = ['NT', 'G', 'FT']
species_labels = ['NANOG', 'GATA6', 'FGF4']
_wait = [0] # [0, 4, 8, 12, 16, 24, 32, 40]
rows = 1 # len(_wait)
cols = 1 # len(species_selection)
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[0, 0]
pick = (369, 50) # ('trajectory', 'cell') # {(369, 49)}
for wait in _wait:
    row = 0 # _wait.index(wait)
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, wait, verbose = True)
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        initiate_set_keys = list(initiate_set_temp.keys())
        initiate_set_values = list(initiate_set_temp.values())
        initiate_set = list(zip(*initiate_set_values))
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            trajectory_set = data_memo_initiate[(cellulate, initiate)][pick[0], :]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            cellular_stats = objective_fun_cellular_stats(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            classifier = objective_fun_classifier(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            coco = cocos[initiate_index % len(cocos)]
            cap = caps[initiate_index // len(cocos)]
            edge = edges[initiate_index // len(cocos)]
            for spa in species_selection:
                spa_index = species_selection.index(spa)
                col = 0 # species_selection.index(spa)
                y_cellular_stats = cellular_stats[spa]
                if synopses['Mean']:
                    y_mean = y_cellular_stats[0, :, pick[1]].flatten()
                    axe.plot(x, y_mean, color = cocos[spa_index], fillstyle = cap, linestyle = edges[0], marker = 'None', linewidth = 2, label = species_labels[spa_index])
                if synopses['Sand']:
                    y_mean = y_cellular_stats[0, :, pick[1]].flatten()
                    y_sand = np.array([np.std(y_cellular_stats[:, _, :]) for _ in range(y_cellular_stats.shape[1])])
                    axe[row, col].plot(x, y_mean-y_sand, color = cocos[spa_index], fillstyle = cap, linestyle = edges[1], marker = 'None', linewidth = 1, alpha = 0.5)
                    axe[row, col].plot(x, y_mean+y_sand, color = cocos[spa_index], fillstyle = cap, linestyle = edges[1], marker = 'None', linewidth = 1, alpha = 0.5)
                _x = np.array([_ for _ in x if _ % 4 == 0], 'int')
                _y = np.linspace(0, 1000, 11).astype('int')
                # if row == 0: axe[row, col].set_title(label = spa)
                axe.set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                axe.set_yticks(ticks = _y, labels = [_ if _ % 200 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                if row == rows-1: axe.set_xlabel('Time [Hour]', fontsize = font_size_bet)
                if col == 0: axe.set_ylabel('Molecule Count', fontsize = font_size_bet)
                axe.set_xlim(time_mini-time_slit, time_maxi+time_slit) # x_limes
                axe.set_ylim(0, 1000) # y_limes
                # axe[row, col].grid(alpha = 0.125)
                axe.set_title(label = 'Cell Level Example', fontsize = font_size_bet)
                axe.set_title(label = '[A]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
                axe.legend(fontsize = font_size_chi, loc = 'upper left', title = 'Total', title_fontsize = font_size_chi)
# fig.suptitle(t = 'Cellular Stats')

# Visualize Data Memo [Counter Cell]

_initiate_set_keys = ['FC_MRNA', 'FC'] # ['MRNA', 'PRO']
_initiate_set_values = [
    [[0], [100]],
    [[0], [100]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]
data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, initiate_set_temp, 0, verbose = True)

synopses = {'Mean': 1, 'Sand': 0}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (3, 7))]
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
rows = 1
cols = 1 # len(clues) # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[0, 1]
pick = 100
for cellulate in cellulates:
    cellulate_index = cellulates.index(cellulate)
    cellulate_cells = cells[cellulate_index]
    normalization = np.power(cellulate_cells, normalize)
    initiate_set_keys = list(initiate_set_temp.keys())
    initiate_set_values = list(initiate_set_temp.values())
    initiate_set = list(zip(*initiate_set_values))
    for initiate in initiate_set:
        initiate_index = initiate_set.index(initiate)
        trajectory_set = data_memo_initiate[(cellulate, initiate)][pick, :]
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
        counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
        x = objective_fun.tau
        y = counter if not normalize else {key: counter[key]/normalization for key in counter.keys()}
        coco = cocos[initiate_index % len(cocos)]
        cap = caps[initiate_index // len(cocos)]
        edge = edges[initiate_index // len(cocos)]
        row = 0
        col = 0
        for clue in clues:
            clue_index = clues.index(clue)
            if synopses['Mean']:
                if clue != 'U':
                    coco = cocos[clue_index]
                    y_mean = np.mean(y[clue], 0).flatten()
                else:
                    coco = cocos[8]
                    y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                axe.plot(x, y_mean, color = coco, fillstyle = cap, linestyle = edges[0], marker = 'None', linewidth = 2, label = clue_labels[clue_index])
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
            axe.set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
            axe.set_yticks(ticks = _y, labels = [_ if _ % 20 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
            if row == rows-1: axe.set_xlabel('Time [Hour]', fontsize = font_size_bet)
            if col == 0: axe.set_ylabel('Cell Count', fontsize = font_size_bet)
            axe.set_xlim(time_mini-time_slit, time_maxi+time_slit) # x_limes
            axe.set_ylim(0, 100) # y_limes
            # axe[row, col].grid(alpha = 0.125)
        axe.axhline(aim_NT[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'b' if not normalize else cocks[0], linestyle = ':', linewidth = 2, alpha = 0.25, label = 'EPI Target')
        axe.axhline(aim_G[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'r' if not normalize else cocks[1], linestyle = ':', linewidth = 2, alpha = 0.25, label = 'PRE Target')
        axe.set_title(label = 'Tissue Level Example', fontsize = font_size_bet)
        axe.set_title(label = '[B]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
axe.legend(fontsize = font_size_chi, loc = 'upper right')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig3'
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
