#####################################
######## Paper Draft Cartoon ########
#####################################

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

cards = [1, 5, 10, 25, 50, 75, 100] # {1, 5, 10, 25, 50, 75, 100}

_initiate_set_keys = ['N_MRNA', 'G_MRNA', 'N', 'G']
_initiate_set_values = [
    [[200, 175, 150, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 50, 25, 0]],
    [[0, 25, 50, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 150, 175, 200]],
    [[200, 175, 150, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 50, 25, 0]],
    [[0, 25, 50, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 150, 175, 200]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

#%%# Retrieve Data [Initiate]

def retrieve_data_cellulate_card_initiate(memo, cellulate, card, initiate, reiteration, verbose = False):
    from Utilities import simul_data_load
    if verbose: print(f"{' '*8}{cellulate}{' '*8}{initiate}")
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
    tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate_card_initiate = (theta_set, trajectory_set)
    return data_cellulate_card_initiate

def retrieve_data_memo_card_initiate(memo, cellulates, cards, initiate_set, verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    keys = list(initiate_set.keys())
    values = list(initiate_set.values())
    initiate_set_temp = list(zip(*values))
    data_memo_card_initiate = {(cellulate, card, initiate_temp): None for cellulate in cellulates for card in cards for initiate_temp in initiate_set_temp}
    for cellulate in cellulates:
        for card in cards:
            for initiate_temp in initiate_set_temp:
                initiate = {keys[index]: initiate_temp[index] for index in range(len(keys))}
                _initiate = '_'.join(map(str, [f'{key}_{value}' for key, value in initiate.items()]))
                reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}_Card_{card}_Initiate_{_initiate}'
                data_cellulate_card_initiate = retrieve_data_cellulate_card_initiate(memo, cellulate, card, initiate, reiteration, verbose)
                data_memo_card_initiate[(cellulate, card, initiate_temp)] = data_cellulate_card_initiate[1]
    return data_memo_card_initiate

#%%# Visualize Data Memo [Counter]

cards = [1, 5, 10, 25, 50, 75, 100] # {1, 5, 10, 25, 50, 75, 100}

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

synopses = {'Mean': 1, 'Sand': 0, 'Quantiles': 0}
quantiles = [0.5] # {[], [0.5], [0.25, 0.5, 0.75], [0.025, 0.5, 0.975]}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (3, 7))]
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND'] # ['EPI Fate', 'PRE Fate', 'UND Fate']
rows = 1
cols = len(clues) # {'NT', 'G'}
epoch = 48*int(1/time_delta)
initiate_set_keys = list(initiate_set_temp.keys())
initiate_set_values = list(initiate_set_temp.values())
initiate_set = list(zip(*initiate_set_values))
mat = np.full((len(clues), len(cards), len(initiate_set)), np.nan)
aims = [aim_NT[1], aim_G[1]]
_aims = [36.48, 60.91, 100-(36.48+60.91)]
for cellulate in cellulates:
    cellulate_index = cellulates.index(cellulate)
    cellulate_cells = cells[cellulate_index]
    normalization = np.power(cellulate_cells, normalize)
    for card in cards:
        card_index = cards.index(card)
        _card = [card]
        data_memo_card_initiate = retrieve_data_memo_card_initiate(memo, cellulates, _card, initiate_set_temp, verbose = True)
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            trajectory_set = data_memo_card_initiate[(cellulate, card, initiate)]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            y = counter if not normalize else {key: counter[key]/normalization for key in counter.keys()}
            coco = cocos[initiate_index % len(cocos)]
            cap = caps[card_index % len(caps)]
            edge = edges[card_index % len(edges)]
            for row in range(rows):
                for col in range(cols):
                    clue = clues[col]
                    if synopses['Mean']:
                        if clue != 'U':
                            y_mean = np.mean(y[clue], 0).flatten()
                        else:
                            y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                        mat[col, card_index, initiate_index] = np.round(y_mean[epoch] - _aims[col]) # y_mean[epoch] - aims[col]*cellulate_cells/normalization
                    if synopses['Sand']:
                        y_sand = np.std(y[clue], 0).flatten()
                        y_net = y_mean-y_sand
                        y_pot = y_mean+y_sand
                    elif synopses['Quantiles']:
                        pass

#%%# Create Figure!

rows = 1
cols = 4
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)

# Visualize Data Memo [Counter Too]

_initiate_set_keys = ['N_MRNA', 'G_MRNA', 'N', 'G']
_initiate_set_values = [
    [[200, 175, 150, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 50, 25, 0]],
    [[0, 25, 50, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 150, 175, 200]],
    [[200, 175, 150, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 50, 25, 0]],
    [[0, 25, 50, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 150, 175, 200]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

initiate_set_keys = list(initiate_set_temp.keys())
initiate_set_values = list(initiate_set_temp.values())
initiate_set = list(zip(*initiate_set_values))

import seaborn as sns
sns.reset_orig()

# cmap = sns.color_palette(palette = 'RdBu_r', as_cmap = True).reversed() # {'Spectral', 'RdBu', 'coolwarm', 'bwr'}
cmap = sns.diverging_palette(h_neg = 310, h_pos = 170, s = 100, l = 50, sep = 10, n = 100, center = 'light', as_cmap = True)
# sns.set(font_scale = 1)
verbose = False
mat[mat == 0] = 0
rows = 1
cols = len(clues) # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
row = 0
_x = [f'{_[1]-100}%' for _ in initiate_set]
_y = cards
tits = ['[F]', '[G]', '[H]']
axe = axe_mat[0:1, 1:4]
for col in range(cols):
    cbar = True if col == cols - 1 else False
    heat = mat[col]
    annotate = np.rint(heat) if verbose else False
    mini = -1*aims[col]*cellulate_cells/normalization if col < 2 else -50
    maxi = cellulate_cells/normalization-aims[col]*cellulate_cells/normalization if col < 2 else 50
    heat_map = sns.heatmap(data = heat, vmin = mini, vmax = maxi, cmap = cmap, center = 0, annot = annotate, fmt = '.0f', annot_kws = {'fontsize': font_size_chi}, square = False, xticklabels = _x, yticklabels = _y, cbar = cbar, linewidth = 0.25, ax = axe[row, col])
    heat_map.invert_yaxis()
    axe[row, col].set_title(label = clue_labels[col], y = 1, fontsize = font_size_bet)
    axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    axe[row, col].set_xlabel(xlabel = 'Initial Condition Perturbation (ICP)', fontsize = font_size_bet)
    axe[row, 0].set_ylabel(ylabel = 'Number of Perturbed Cells', fontsize = font_size_bet)
    heat_map.set_xticklabels(_x, size = font_size_bet)
    heat_map.set_yticklabels(_y if col == 0 else [], size = font_size_bet)
    # heat_map.collections[0].colorbar.ax.tick_params(labelsize = font_size)
    if col == cols-1:
        ticks = np.linspace(mini, maxi, 11).astype('int')
        labels = [_ if _ % 20 == 10 or _ == 0 else None for _ in ticks] # if col != cols-1 else [_ if _ % 20 == 10 or _ == 0 else None for _ in ticks]
        heat_map.collections[0].colorbar.ax.set_yticks(ticks = ticks, labels = labels, fontsize = font_size_bet)
        heat_map.collections[0].colorbar.ax.set_ylabel(ylabel = 'Fate Error (FE)', fontsize = font_size_bet) # Fate Deviation (FD)
# fig.suptitle(t = f"{' ~ '.join(initiate_set_keys)}") # Standardized Absolute Fate Deviation (SAFD)
# plt.show()

# Visualize Data Memo [Initial Condition]

_initiate_set_keys = ['N_MRNA', 'G_MRNA', 'N', 'G']
_initiate_set_values = [
    [[200, 150, 100, 50, 0][::-1]],
    [[0, 50, 100, 150, 200][::-1]],
    [[200, 150, 100, 50, 0][::-1]],
    [[0, 50, 100, 150, 200][::-1]]
]
percentage = [0, 1] # toll = [1, 0]
_initiate_set = [{_initiate_set_keys[i]: _initiate_set_values[i][j] for i in range(len(_initiate_set_keys))} for j in range(len(data_paths))]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]

data_memo_card_initiate = retrieve_data_memo_card_initiate(memo, cellulates, [100], initiate_set_temp, verbose = True)

synopses = {'Mean': 1, 'Sand': 0}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
cocos = ['limegreen', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:brown'] # ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# cocks = ['c', 'g', 'm', 'b', 'r', 'y', 'k', 'w']
marks = ['s', 'D', 'o', '*', 'X']
_species_selection = [['N', 'G']] # [['N', 'G'], ['N_MRNA', 'G_MRNA']]
_species_selection_labels = [['NANOG', 'GATA6']] # [['NANOG', 'GATA6'], ['NANOG mRNA', 'GATA6 mRNA']]
_species_selection_limes = [(0, 1000), (0, 250)]
rows = 1
cols = 1 # len(clues) # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
mark_size = 11
axe = axe_mat[0:1, 0:1]
_axe_x = np.empty_like(axe)
_axe_x[0, 0] = axe[0, 0].inset_axes([0, 1, 1, 0.2])
_axe_y = np.empty_like(axe)
_axe_y[0, 0] = axe[0, 0].inset_axes([1, 0, 0.2, 1])
for cellulate in cellulates:
    cellulate_index = cellulates.index(cellulate)
    cellulate_cells = cells[cellulate_index]
    normalization = np.power(cellulate_cells, normalize)
    initiate_set_keys = list(initiate_set_temp.keys())
    initiate_set_values = list(initiate_set_temp.values())
    initiate_set = list(zip(*initiate_set_values))
    for initiate in initiate_set:
        initiate_index = initiate_set.index(initiate)
        trajectory_set = data_memo_card_initiate[(cellulate, card, initiate)]
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
        objective_fun.apply(tau_mini = time_mini, tau_maxi = time_maxi)
        t = objective_fun.tau
        z = objective_fun.data
        coco = cocos[initiate_index % len(cocos)]
        mark = marks[initiate_index % len(cocos)]
        for species_selection in _species_selection:
            species_selection_index = _species_selection.index(species_selection)
            species_selection_labels = _species_selection_labels[species_selection_index]
            species_selection_limes = _species_selection_limes[species_selection_index]
            row = 0
            col = species_selection_index
            x_index = species.index(species_selection[0])
            y_index = species.index(species_selection[1])
            x = z[:, x_index, 0, :].flatten()
            y = z[:, y_index, 0, :].flatten()
            axe[row, col].fill_between([0, 1000], 0, 1000, color = 'tab:olive', alpha = 0.025)
            axe[row, col].scatter(x, y, s = mark_size, c = coco, marker = mark, edgecolors = 'none', alpha = 0.75, label = f'{initiate[1]-100}%')
            hoax = [200+5*np.sqrt(200), 1000-5*np.sqrt(1000)]
            axe[row, col].hlines(hoax, [500-5*np.sqrt(500), 0], [1000, 200+5*np.sqrt(200)], colors = 'tab:olive', linestyles = '-', linewidths = 1, label = None)
            axe[row, col].hlines(hoax, [500-5*np.sqrt(500), 0], [1000, 200+5*np.sqrt(200)], colors = ['tab:blue', 'tab:orange'], linestyles = '--', linewidths = 1, label = None)
            axe[row, col].fill_between([0, 200+5*np.sqrt(200)], 1000-5*np.sqrt(1000), 1000, color = 'tab:orange', alpha = 0.05)
            very = [200+5*np.sqrt(200), 500-5*np.sqrt(500)]
            axe[row, col].vlines(very, [1000-5*np.sqrt(1000), 0], [1000, 200+5*np.sqrt(200)], colors = 'tab:olive', linestyles = '-', linewidths = 1, label = None)
            axe[row, col].vlines(very, [1000-5*np.sqrt(1000), 0], [1000, 200+5*np.sqrt(200)], colors = ['tab:orange', 'tab:blue'], linestyles = '--', linewidths = 1, label = None)
            axe[row, col].fill_between([500-5*np.sqrt(500), 1000], 0, 200+5*np.sqrt(200), color = 'tab:blue', alpha = 0.05)
            _x = np.linspace(species_selection_limes[0], species_selection_limes[1], 11).astype('int')
            _y = np.linspace(species_selection_limes[0], species_selection_limes[1], 11).astype('int')
            axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 200 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 200 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_xlabel(species_selection_labels[0], fontsize = font_size_bet)
            axe[row, col].set_ylabel(species_selection_labels[1], fontsize = font_size_bet)
            axe[row, col].set_xlim(species_selection_limes[0], species_selection_limes[1]) # x_limes
            axe[row, col].set_ylim(species_selection_limes[0], species_selection_limes[1]) # y_limes
            if initiate in [(0, 200, 0, 200), (200, 0, 200, 0)]:
                nome = 10
                weights = 0.05*np.ones(tuple([len(x)]))
            else:
                nome = 1
                weights = None
            _axe_x[row, col].hist(x, bins = int((species_selection_limes[1]-species_selection_limes[0])/nome), range = (species_selection_limes[0], species_selection_limes[1]), density = False, weights = weights, histtype = 'stepfilled', color = coco, alpha = 0.75)
            _axe_y[row, col].hist(y, bins = int((species_selection_limes[1]-species_selection_limes[0])/nome), range = (species_selection_limes[0], species_selection_limes[1]), density = False, weights = weights, histtype = 'stepfilled', orientation = 'horizontal', color = coco, alpha = 0.75)
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
            axe[row, col].set_title(label = 'Initial Condition Distribution (ICD)', loc = 'center', fontsize = font_size_bet)
            axe[row, col].set_title(label = '[E]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold', position = (-0.25, 1))
            axe[row, col].legend(loc = 'upper right', fontsize = font_size_chi, borderpad = 0.25) # axe[row, col].legend(bbox_to_anchor = (1, 1), loc = 'lower left', fontsize = font_size_chi)
            axe[row, col].set_aspect(aspect = 1)
            axe[row, col].text(x = (1000+500-5*np.sqrt(500))/2, y = (200+5*np.sqrt(200))/2, s = 'EPI', alpha = 1, color = 'tab:blue', fontsize = font_size_bet, horizontalalignment = 'center', verticalalignment = 'center')
            axe[row, col].text(x = (200+5*np.sqrt(200))/2, y = (1000+1000-5*np.sqrt(1000))/2, s = 'PRE', alpha = 1, color = 'tab:orange', fontsize = font_size_bet, horizontalalignment = 'center', verticalalignment = 'center')
            axe[row, col].text(x = 1000*(1-1/4-1/8), y = 1000*(1-1/4-1/8), s = 'UND', alpha = 1, color = 'tab:olive', fontsize = font_size_bet, horizontalalignment = 'center', rotation = 45, rotation_mode = 'anchor', verticalalignment = 'center')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig9_Bet'
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
