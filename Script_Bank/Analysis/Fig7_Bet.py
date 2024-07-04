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

from Fig0_Services import retrieve_mapes, retrieve_para_set_truths, retrieve_data_memo_auto_para_mem
from Fig0_Services import objective_fun_classifier, objective_fun_cellular_stats
from Fig0_Services import make_kernel

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
_auto_para_mem_set = [7] # Decimals # 8 Instances
auto_para_mem_set_temp = [tuple(map(int, format(d, 'b'))) for d in _auto_para_mem_set] # Binaries
auto_para_mem_set = [tuple([0]*(3-len(tup)))+tup for tup in auto_para_mem_set_temp]
auto_para_mem_rules = [0 if _ == 0 else 1 for _ in _auto_para_mem_set]

#%%# Visualize Data Memo [Classification]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
meth = ('Auto', 'Para', 'Mem') # {('Auto', 'Para', 'Mem'), ('Autocrine', 'Paracrine', 'Membrane')}
data_memo_auto_para_mem = retrieve_data_memo_auto_para_mem(memo, cellulates, auto_para_mem_set, meth, verbose = True)
w = dict()
v = dict()
tau_mini = 0
tau_maxi = 48
tau_delta = 0.25
spa = 'FT' # {'NT', 'G', 'FT'}
spa_label = 'FGF4' # {'NANOG', 'GATA6', 'FGF4'}

for cellulate in cellulates:
    for rule in rules:
        rule_index = rules.index(rule)
        trajectory_set = data_memo_auto_para_mem[(cellulate, auto_para_mem_set[0])]
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, rules = rules, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        classifier = objective_fun_classifier(objective_fun, tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
        x = objective_fun.tau
        y = classifier
        w.update({cellulate: y})
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, rules = rules, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        cellular_stats = objective_fun_cellular_stats(objective_fun, tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
        # x = objective_fun.tau
        y = cellular_stats[spa]
        v.update({cellulate: y})

#%%# Visualize Data Memo [Convolution]

cellulate = cellulates[0]
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
positional_information = False
verbose = True

from scipy.ndimage import convolve

_degrees = [tuple(range(degree+1)) for degree in range(cellulates[0][0])] # [tuple([degree]) for degree in range(cellulates[0][0])]
storage_hood_port_alter = dict()

for degrees in _degrees:
    
    # degrees = (0, 1, 2, 3)
    print(f"{'>'*8} Degrees! {' · '.join(map(str, degrees))} {'<'*8}")
    kernel = make_kernel(degrees, verbose)
    _crux = np.ones(cellulate, int)
    crux = convolve(input = _crux, weights = kernel, mode = 'constant')
    
    neighborhood_quota = {key: None for key in clues}
    hints = clues
    hint_labels = clue_labels
    
    y = w[cellulate]
    u = v[cellulate]
    
    for clue in clues:
        print(f"{'~'*8} Process! {clue_labels[clues.index(clue)]} ({clue}) {'~'*8}")
        shape = (y[clue].shape[0], y[clue].shape[1], cellulate[0], cellulate[1])
        share = np.full(shape, np.nan)
        iota = range(y[clue].shape[0]) # simulation_index
        jota = range(y[clue].shape[1]) # tau_index
        for i in iota:
            for j in jota:
                zone = np.reshape(y[clue][i, j], cellulate)
                neighborhood = np.reshape(u[i, j], cellulate)
                # print(zone)
                # print(neighborhood)
                convolution = convolve(input = neighborhood, weights = kernel, mode = 'constant') # quota
                _neighbor_count = zone*convolution
                cone = [zone == 1]
                chow = [_neighbor_count]
                neighbor_count = np.select(cone, chow, np.nan)
                neighbor_quota = neighbor_count/crux
                # print(neighbor_count)
                share[i, j, ...] = neighbor_quota
        neighborhood_quota.update({clue: share})
    
    # Visualize Data Memo [Probability]
    
    hood_port_stats = ['mean', 'sand', 'cove']
    hood_port_shape = (len(clues), 1, y['U'].shape[1], len(hood_port_stats))
    hood_port_alter = np.full(hood_port_shape, np.nan)
    
    if positional_information:
        pass
    else:
        for clue in clues:
            clue_index = clues.index(clue)
            iota = range(y[clue].shape[0])
            jota = range(y[clue].shape[1])
            simulation_index = slice(min(iota), max(iota)+1) # slice(min(iota), max(iota)+1)
            tau_index = slice(min(jota), max(jota)+1) # slice(min(jota), max(jota)+1)
            alp = neighborhood_quota[clue][:, simulation_index, tau_index, ...]
            print(f"{'~'*8} Storage! {alp.shape} {clue_labels[clues.index(clue)]} ({clue}) {'~'*8}")
            for _ in range(alp.shape[1]):
                bet = np.nanmean(alp[:, _, ...])
                chi = np.nanstd(alp[:, _, ...])
                dev = chi/bet
                hood_port_alter[clue_index, 0, _, :] = [bet, chi, dev]
                # print(x[_], bet, chi, dev)
    
    storage_hood_port_alter.update({degrees: hood_port_alter})

#%%# Create Figure Zero!

rows = 1
cols = len(clues)
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)

synopses = {'Mean': 1, 'Sand': 1, 'Quantiles': 0}
cocks = list(matplotlib.colors.TABLEAU_COLORS.keys())
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[0:rows+1, 0:cols+1]
tits = ['[A]', '[B]', '[C]']
x_limes = (np.min(x), np.max(x))
y_limes = (0, 120)
_x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
_y = np.linspace(y_limes[0], y_limes[1], 13).astype('int')
medal = ''

for clue in clues:
    clue_index = clues.index(clue)
    row = clue_index // cols
    col = clue_index % cols
    for degrees in _degrees:
        degrees_index = _degrees.index(degrees)
        cock = cocks[degrees_index]
        if synopses['Mean']:
            medal = r'$\mu$'
            y_mean_alter = storage_hood_port_alter[degrees][clue_index, 0, :, hood_port_stats.index('mean')]
            axe[row, col].plot(x, y_mean_alter, color = cock, linestyle = '-', linewidth = 2, label = '·'.join(map(str, degrees)), alpha = 1)
        if synopses['Mean'] and synopses['Sand']:
            medal = r'$\mu \pm \sigma$'
            y_sand_alter = storage_hood_port_alter[degrees][clue_index, 0, :, hood_port_stats.index('sand')]
            axe[row, col].plot(x, y_mean_alter-y_sand_alter, color = cock, linestyle = '-.', linewidth = 1, label = None, alpha = 0.5)
            axe[row, col].plot(x, y_mean_alter+y_sand_alter, color = cock, linestyle = '-.', linewidth = 1, label = None, alpha = 0.5)
        if synopses['Quantiles']:
            pass
        axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
        axe[row, col].set_ylabel(spa_label if col == 0 else None, fontsize = font_size_bet)
        x_tick_labels = [_ if _ % 8 == 0 else None for _ in _x] if row == rows-1 else [None]*len(_x)
        y_tick_labels = [_ if _ % 20 == 0 else None for _ in _y] if col == 0 else [None]*len(_y)
        axe[row, col].set_xticks(ticks = _x, labels = x_tick_labels, fontsize = font_size_bet, rotation = 0)
        axe[row, col].set_yticks(ticks = _y, labels = y_tick_labels, fontsize = font_size_bet, rotation = 0)
        axe[row, col].set_xlim(x_limes)
        axe[row, col].set_ylim(y_limes)
    if clue == 'U': axe[row, col].legend(fontsize = font_size_chi, loc = 'upper left', ncols = 2)
    axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    axe[row, col].set_title(label = clue_labels[col] + '\n' + medal, fontsize = font_size_bet)

#%%# Create Figure One!

rows = 1
cols = 1
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)

synopses = {'Mean': 1, 'Sand': 1, 'Quantiles': 0}
cocos = ['tab:blue', 'tab:orange', 'tab:olive'] # list(matplotlib.colors.TABLEAU_COLORS.keys())
marks = ['o', 'D', 's']
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
axe = axe_mat[0:rows+1, 0:cols+1]
tits = ['[E]']
d_limes = (-0.5, len(_degrees)-0.5)
y_limes = (0, 120)
d = np.linspace(0, len(_degrees)-1, len(_degrees)).astype('int')
_d = d
_y = np.linspace(y_limes[0], y_limes[1], 13).astype('int')
z = {clue: np.full((len(_degrees), len(hood_port_stats)), np.nan) for clue in clues}
d_label = 'Max Neighborhood Degree' # {'Neighborhood Degree', 'Max Neighborhood Degree'}
epoch = 48*int(1/time_delta)

for clue in clues:
    clue_index = clues.index(clue)
    for degrees in _degrees:
        degrees_index = _degrees.index(degrees)
        z[clue][degrees_index, :] = storage_hood_port_alter[degrees][clue_index, 0, epoch, :]

row = 0
col = 0
if synopses['Mean'] and synopses['Sand']:
    _y_limes = (0, 40)
    _y_ticks = np.linspace(_y_limes[0], _y_limes[1], 5).astype('int')
    _axe = axe[row, col].inset_axes([0.45, 0.6, 0.5, 0.3])

for clue in clues:
    clue_index = clues.index(clue)
    coco = cocos[clue_index]
    mark = marks[clue_index]
    if synopses['Mean']:
        y_mean_alter = z[clue][:, hood_port_stats.index('mean')]
        axe[row, col].plot(d, y_mean_alter, color = coco, linestyle = '-', linewidth = 2, marker = mark, label = clue_labels[clue_index], alpha = 0.75)
    if synopses['Mean'] and synopses['Sand']:
        y_sand_alter = z[clue][:, hood_port_stats.index('sand')]
        _axe.plot(d, y_sand_alter, color = coco, linestyle = '--', linewidth = 1, marker = mark, markersize = 5, label = clue_labels[clue_index], alpha = 0.5)
        _axe.set_xticks(ticks = _d, labels = _d, fontsize = font_size_chi, rotation = 0)
        _axe.set_yticks(ticks = _y_ticks, labels = [_ if _ % 20 == 0 else None for _ in _y_ticks], fontsize = font_size_chi, rotation = 0)
        _axe.set_xlim(d_limes)
        _axe.set_ylim(_y_limes)
        _axe.set_title(label = r'$\sigma$', fontsize = font_size_bet)
    if synopses['Quantiles']:
        pass
    axe[row, col].set_xlabel(d_label, fontsize = font_size_bet)
    axe[row, col].set_ylabel(spa_label if col == 0 else None, fontsize = font_size_bet)
    axe[row, col].set_xticks(ticks = _d, labels = _d, fontsize = font_size_bet, rotation = 0)
    axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 20 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
    axe[row, col].set_xlim(d_limes)
    axe[row, col].set_ylim(y_limes)
    if clue == 'U': axe[row, col].legend(fontsize = font_size_chi, loc = 'lower right', ncols = 1)
    axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    axe[row, col].set_title(label = r'$\mu$' + ' ' + f'(Time = {int(j*time_delta)} Hours)', fontsize = font_size_bet)

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig7_Bet'
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
