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
fig_resolution = 750 # {250, 375, 500, 750, 1000}
plt.rcParams['figure.dpi'] = fig_resolution
plt.rcParams['savefig.dpi'] = fig_resolution
fig_size_base = 3.75 # {2.5, 3.75, 5, 7.5, 10}

#%%# Data Preparation

from Fig0_Services import retrieve_mapes, retrieve_para_set_truths, retrieve_data_memo_initiate
from Fig0_Services import objective_fun_cellular_stats, objective_fun_classifier, objective_fun_counter
from Fig0_Services import make_tau_tally

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
cols = 3
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = True)

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

synopses = {'Mean': 1, 'Sand': 1}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
edges = ['-', '--']
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
species_selection = ['NT', 'G', 'FT']
species_labels = ['NANOG', 'GATA6', 'FGF4']
_wait = [0] # [0, 4, 8, 12, 16, 24, 32, 40]
rows = 1
cols = len(clues)
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[0:1, 0:3]
x_limes = (time_mini-time_slit, time_maxi+time_slit)
y_limes = (0, 1000)
cell_tally = 1
epoch = 48*int(1/time_delta)
tits = ['[A]', '[B]', '[C]']
for wait in _wait:
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
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            edge = edges[initiate_index // len(cocos)]
            for clue in clues:
                clue_index = clues.index(clue)
                row = 0
                col = clue_index
                y_classifier = classifier[clue]
                tau_tally = make_tau_tally(counter[clue], cell_tally, x, correct = None, verbose = False)
                v = {'Mini': np.min(tau_tally), 'Mean': np.mean(tau_tally), 'Maxi': np.max(tau_tally)} # v = np.min(np.argwhere(a = y_classifier)[:, 1])
                y_classifier_epoch = np.repeat(y_classifier[:, [epoch], :], len(x), 1)
                for spa in species_selection:
                    spa_index = species_selection.index(spa)
                    y_cellular_stats = cellular_stats[spa]
                    y = np.ma.masked_array(data = y_cellular_stats, mask = np.logical_not(y_classifier_epoch))
                    if synopses['Mean']:
                        y_mean = np.mean(np.mean(y, 0), 1)
                        coco = cocos[spa_index % len(cocos)]
                        axe[row, col].plot(x, y_mean, color = coco, linestyle = edge, marker = 'None', linewidth = 2, label = species_labels[spa_index])
                    if synopses['Sand']:
                        y_mean = np.mean(np.mean(y, 0), 1)
                        y_sand = np.array([np.std(y[:, _, :]) for _ in range(y_cellular_stats.shape[1])])
                        axe[row, col].plot(x, y_mean-y_sand, color = cocos[spa_index], linestyle = edges[1], marker = 'None', linewidth = 1, alpha = 0.5)
                        axe[row, col].plot(x, y_mean+y_sand, color = cocos[spa_index], linestyle = edges[1], marker = 'None', linewidth = 1, alpha = 0.5)
                    axe[row, col].fill_between([v['Mini'], v['Maxi']], y_limes[0], y_limes[1], color = 'tab:gray', alpha = 0.075)
                    axe[row, col].axvline(x = v['Mean'], ymin = 0, ymax = 1, linestyle = '-.', linewidth = 1, color = 'tab:gray', alpha = 0.75)
                    _x = np.array([_ for _ in x if _ % 4 == 0], 'int')
                    _y = np.linspace(y_limes[0], y_limes[1], 9).astype('int')
                    axe[row, col].set_title(label = tits[clue_index], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
                    axe[row, col].set_title(label = f'{clue_labels[clue_index]} μ ± σ', fontsize = font_size_bet)
                    axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                    axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 250 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0) if col != cols else axe[row, col].set_yticks(ticks = _y, labels = [], fontsize = font_size_alp, rotation = 0)
                    if col != cols: axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
                    if col != cols: axe[row, col].set_ylabel('Molecule Count', fontsize = font_size_bet)
                    axe[row, col].set_xlim(x_limes[0], x_limes[1]) # x_limes
                    axe[row, col].set_ylim(y_limes[0], y_limes[1]) # y_limes
                    if col == 0: axe[row, col].legend(fontsize = font_size_chi, loc = 'upper right', title = 'Total', title_fontsize = font_size_chi)
# plt.show()

# Visualize Data Memo [Final Condition]

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

normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
cocos = ['tab:olive', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan'] # ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
cock = 'm'
marks = ['s', 'o', 'D']
mark_size = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23}
clues = ['U', 'NT', 'G']
clue_labels = ['UND', 'EPI', 'PRE']
species_selection = ['NT', 'G', 'FT']
species_selection_labels = ['NANOG', 'GATA6', 'FGF4']
_wait = [0] # [0, 4, 8, 12, 16, 24, 32, 40]
rows = 1
cols = 3
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[1:2, 0:3]
_axe_h = np.empty_like(axe)
_axe_h[0, 0] = axe[0, 0].inset_axes([0, 1, 1, 0.2])
_axe_h[0, 1] = axe[0, 1].inset_axes([0, 1, 1, 0.2])
_axe_h[0, 2] = axe[0, 2].inset_axes([0, 1, 1, 0.2])
_axe_v = np.empty_like(axe)
_axe_v[0, 0] = axe[0, 0].inset_axes([1, 0, 0.2, 1])
_axe_v[0, 1] = axe[0, 1].inset_axes([1, 0, 0.2, 1])
_axe_v[0, 2] = axe[0, 2].inset_axes([1, 0, 0.2, 1])
x_limes = (0, 750) # NT
y_limes = (0, 1250) # G
z_limes = (0, 250) # FT
weights = None
tits = ['[D]', '[E]', '[F]']
for wait in _wait:
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
            t = objective_fun.tau
            for clue in clues:
                clue_index = clues.index(clue)
                _classifier = classifier[clue]
                mark = marks[clue_index]
                x_cellular_stats = cellular_stats['NT']
                y_cellular_stats = cellular_stats['G']
                z_cellular_stats = cellular_stats['FT']
                x_temp = np.ma.masked_array(data = x_cellular_stats, mask = np.logical_not(_classifier))
                y_temp = np.ma.masked_array(data = y_cellular_stats, mask = np.logical_not(_classifier))
                z_temp = np.ma.masked_array(data = z_cellular_stats, mask = np.logical_not(_classifier))
                x = x_temp[:, -1, :].flatten()
                y = y_temp[:, -1, :].flatten()
                z = z_temp[:, -1, :].flatten()
                coco = cocos[clue_index % len(cocos)]
                _x = np.linspace(0, 1250, 11).astype('int')
                _y = np.linspace(0, 1250, 11).astype('int')
                _z = np.linspace(0, 1250, 51).astype('int')
                row = 0
                col = 0
                axe[row, col].scatter(x, y, s = mark_size, c = coco, marker = mark, edgecolors = 'none', alpha = 0.75, label = clue_labels[clue_index])
                x_mat = np.vstack([x, np.ones(len(x))]).T
                slope_x_y, y_cross = np.linalg.lstsq(x_mat, y, rcond = None)[0]
                rest_x_y = np.corrcoef(x, y)[0, 1]
                where_x_y = ((x_limes[1]-x_limes[0])/2, (y_limes[1]-y_limes[0])/2)
                axe[row, col].axline(xy1 = (0, y_cross), slope = slope_x_y, linestyle = '--', linewidth = 1, color = cock)
                axe[row, col].annotate(text = r'$r_{xy}$' + ' = ' + f'{np.round(rest_x_y, 3)}', xy = where_x_y, xytext = where_x_y, color = cock, fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
                axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
                axe[row, col].set_title(label = 'Time = 48 Hours', fontsize = font_size_bet)
                axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 250 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 250 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_xlabel(species_selection_labels[0], fontsize = font_size_bet)
                axe[row, col].set_ylabel(species_selection_labels[1], fontsize = font_size_bet)
                axe[row, col].set_xlim(x_limes[0], x_limes[1]) # x_limes
                axe[row, col].set_ylim(y_limes[0], y_limes[1]) # y_limes
                axe[row, col].legend(fontsize = font_size_chi, loc = 'upper right')
                row = 0
                col = 1
                axe[row, col].scatter(x, z, s = mark_size, c = coco, marker = mark, edgecolors = 'none', alpha = 0.75, label = clue_labels[clue_index])
                x_mat = np.vstack([x, np.ones(len(x))]).T
                slope_x_z, z_cross = np.linalg.lstsq(x_mat, z, rcond = None)[0]
                rest_x_z = np.corrcoef(x, z)[0, 1]
                where_x_z = ((x_limes[1]-x_limes[0])/4, (z_limes[1]-z_limes[0])/2)
                axe[row, col].axline(xy1 = (0, z_cross), slope = slope_x_z, linestyle = '--', linewidth = 1, color = cock)
                axe[row, col].annotate(text = r'$r_{xy}$' + ' = ' + f'{np.round(rest_x_z, 3)}', xy = where_x_z, xytext = where_x_z, color = cock, fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'center', verticalalignment = 'center')
                axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
                axe[row, col].set_title(label = 'Time = 48 Hours', fontsize = font_size_bet)
                axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 250 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_yticks(ticks = _z, labels = [_ if _ % 50 == 0 else None for _ in _z], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_xlabel(species_selection_labels[0], fontsize = font_size_bet)
                axe[row, col].set_ylabel(species_selection_labels[2], fontsize = font_size_bet)
                axe[row, col].set_xlim(x_limes[0], x_limes[1]) # x_limes
                axe[row, col].set_ylim(z_limes[0], z_limes[1]) # z_limes
                # axe[row, col].legend(fontsize = font_size_chi, loc = 'center')
                row = 0
                col = 2
                axe[row, col].scatter(z, y, s = mark_size, c = coco, marker = mark, edgecolors = 'none', alpha = 0.75, label = clue_labels[clue_index])
                z_mat = np.vstack([z, np.ones(len(z))]).T
                slope_z_y, y_cross = np.linalg.lstsq(z_mat, y, rcond = None)[0]
                rest_z_y = np.corrcoef(z, y)[0, 1]
                where_z_y = ((z_limes[1]-z_limes[0])/2, (y_limes[1]-y_limes[0])/2)
                axe[row, col].axline(xy1 = (0, y_cross), slope = slope_z_y, linestyle = '--', linewidth = 1, color = cock)
                axe[row, col].annotate(text = r'$r_{xy}$' + ' = ' + f'{np.round(rest_z_y, 3)}', xy = where_z_y, xytext = where_z_y, color = cock, fontsize = font_size_bet, fontweight = 'book', horizontalalignment = 'center', verticalalignment = 'center')
                axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
                axe[row, col].set_title(label = 'Time = 48 Hours', fontsize = font_size_bet)
                axe[row, col].set_xticks(ticks = _z, labels = [_ if _ % 50 == 0 else None for _ in _z], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 250 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
                axe[row, col].set_xlabel(species_selection_labels[2], fontsize = font_size_bet)
                axe[row, col].set_ylabel(species_selection_labels[1], fontsize = font_size_bet)
                axe[row, col].set_xlim(z_limes[0], z_limes[1]) # z_limes
                axe[row, col].set_ylim(y_limes[0], y_limes[1]) # y_limes
                # axe[row, col].legend(fontsize = font_size_chi, loc = 'center')
            _axe_h[0, 0].hist(x, bins = x_limes[1]-x_limes[0], range = (x_limes[0], x_limes[1]), density = False, weights = weights, histtype = 'stepfilled', color = 'tab:blue', alpha = 0.75)
            _axe_h[0, 0].set_xticks(ticks = _x, labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_h[0, 0].set_yticks(ticks = [], labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_h[0, 0].set_xlim(x_limes[0], x_limes[1])
            _axe_v[0, 0].hist(y, bins = y_limes[1]-y_limes[0], range = (y_limes[0], y_limes[1]), density = False, weights = weights, histtype = 'stepfilled', orientation = 'horizontal', color = 'tab:orange', alpha = 0.75)
            _axe_v[0, 0].set_xticks(ticks = [], labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_v[0, 0].set_yticks(ticks = _y, labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_v[0, 0].set_ylim(y_limes[0], y_limes[1])
            _axe_h[0, 1].hist(x, bins = x_limes[1]-x_limes[0], range = (x_limes[0], x_limes[1]), density = False, weights = weights, histtype = 'stepfilled', color = 'tab:blue', alpha = 0.75)
            _axe_h[0, 1].set_xticks(ticks = _x, labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_h[0, 1].set_yticks(ticks = [], labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_h[0, 1].set_xlim(x_limes[0], x_limes[1])
            _axe_v[0, 1].hist(z, bins = z_limes[1]-z_limes[0], range = (z_limes[0], z_limes[1]), density = False, weights = weights, histtype = 'stepfilled', orientation = 'horizontal', color = 'tab:green', alpha = 0.75)
            _axe_v[0, 1].set_xticks(ticks = [], labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_v[0, 1].set_yticks(ticks = _z, labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_v[0, 1].set_ylim(z_limes[0], z_limes[1])
            _axe_h[0, 2].hist(z, bins = z_limes[1]-z_limes[0], range = (z_limes[0], z_limes[1]), density = False, weights = weights, histtype = 'stepfilled', color = 'tab:green', alpha = 0.75)
            _axe_h[0, 2].set_xticks(ticks = _z, labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_h[0, 2].set_yticks(ticks = [], labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_h[0, 2].set_xlim(z_limes[0], z_limes[1])
            _axe_v[0, 2].hist(y, bins = y_limes[1]-y_limes[0], range = (y_limes[0], y_limes[1]), density = False, weights = weights, histtype = 'stepfilled', orientation = 'horizontal', color = 'tab:orange', alpha = 0.75)
            _axe_v[0, 2].set_xticks(ticks = [], labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_v[0, 2].set_yticks(ticks = _y, labels = [], fontsize = font_size_chi, rotation = 0)
            _axe_v[0, 2].set_ylim(y_limes[0], y_limes[1])
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig4'
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
