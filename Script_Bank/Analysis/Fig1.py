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
from copy import deepcopy # import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
fig_resolution = 500 # {250, 375, 500, 750, 1000}
plt.rcParams['figure.dpi'] = fig_resolution
plt.rcParams['savefig.dpi'] = fig_resolution
fig_size_base = 5 # {2.5, 3.75, 5, 7.5, 10}

#%%# Data Preparation

from Fig0_Services import retrieve_mapes, retrieve_para_set_truths, retrieve_data_memo_auto_para_mem
from Fig1_Services import objective_fun_cellular_stats, objective_fun_classifier, objective_fun_counter

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
_auto_para_mem_set = [0, 7] # Decimals # 8 Instances
auto_para_mem_set_temp = [tuple(map(int, format(d, 'b'))) for d in _auto_para_mem_set] # Binaries
auto_para_mem_set = [tuple([0]*(3-len(tup)))+tup for tup in auto_para_mem_set_temp]
auto_para_mem_rules = [0 if _ == 0 else 1 for _ in _auto_para_mem_set]

#%%# Data Memo [Everything]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
meths = [('Auto', 'Para', 'Mem'), ('Autocrine', 'Paracrine', 'Membrane')] # {('Auto', 'Para', 'Mem'), ('Autocrine', 'Paracrine', 'Membrane')}

lot = 0 # {0, 1}
_meth = '_'.join(map(str, meths[lot]))
data_memo_auto_para_mem = retrieve_data_memo_auto_para_mem(memo, cellulates, auto_para_mem_set, meths[lot], verbose = True)

cellulate = cellulates[0]
tau_mini = 0
tau_maxi = 48
tau_delta = 0.25
pick = (369, 49) # ('trajectory', 'cell') # {(369, 49), (369, 50)}
_container = {rules.index(rule): None for rule in rules}
_colander = {rules.index(rule): None for rule in rules}
_classifier = {rules.index(rule): None for rule in rules}
_counter = {rules.index(rule): None for rule in rules}
_scorer = {rules.index(rule): None for rule in rules}
for rule in rules:
    rule_index = rules.index(rule)
    auto_para_mem = auto_para_mem_set[rule_index]
    trajectory_set = data_memo_auto_para_mem[(cellulate, auto_para_mem)][pick[0], :]
    objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
    cellular_stats = objective_fun_cellular_stats(objective_fun, tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
    _colander[rule_index] = deepcopy(cellular_stats)
    objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
    classifier = objective_fun_classifier(objective_fun, tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
    _classifier[rule_index] = deepcopy(classifier)
    objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
    counter = objective_fun_counter(objective_fun, tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
    _counter[rule_index] = deepcopy(counter)
    objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
    objective_fun.apply(tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
    _scorer[rule_index] = np.copy(objective_fun.data_objective)
    _container[rule_index] = np.copy(objective_fun.data)
x = objective_fun.tau
# y = None

#%%# Create Figure!

rows = 3
cols = 9
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = True)
axe_zero = (1, 0)
verbose = False

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 25 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}

# SBI [Exterminator]

_discard = [[0, 0], [2, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 6], [0, 8]]
for discard in _discard:
    row_discard, col_discard = discard
    axe_mat[row_discard, col_discard].remove()

# SBI [Container]

specs = ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA']
spec_labels = [None]*len(specs)
spec_colors = ['tab:blue', 'tab:orange', 'darkblue', 'darkorange', 'tab:cyan', 'tab:green', 'darkgreen', 'lime', 'olive', 'royalblue', 'deeppink'] # ['lightgray']*len(specs)
line_width_alp = 5 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
line_width_bet = 2 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+1, axe_zero[1]+1+cols)
axe = axe_mat[row_plan, col_plan]
_rows = [min(range(rows)), max(range(rows))]
_cols = [min(range(cols)), max(range(cols))]
x_limes = (time_mini, time_maxi)
y_limes = (-50, 1050)
x_ticks = np.array([_ for _ in x if _ % 4 == 0]).astype('int') if verbose else []
x_tick_labels = [_ if _ % 8 == 0 else None for _ in x_ticks] if verbose else []
y_ticks = np.linspace(y_limes[0], y_limes[1], int((y_limes[1]-y_limes[0])/50)+1).astype('int') if verbose else []
y_tick_labels = [_ if _ % 250 == 0 else None for _ in y_ticks] if verbose else []
_tit = 'A'
tits = np.array([[f'[{_tit+str(rule_index+1)}]'] for rule_index in range(len(rules))])
text_box = {'boxstyle': 'round', 'linestyle': '-.', 'linewidth': line_width_bet, 'pad': 0.25, 'facecolor': 'white', 'edgecolor': 'tab:gray', 'alpha': 0.125}
for rule in rules:
    rule_index = rules.index(rule)
    row = _rows[rule_index]
    col = _cols[rule_index]
    tit = tits[rule_index, col]
    z = _container[rule_index]
    for spec in specs:
        spec_index = specs.index(spec)
        # x = np.arange(0, tau_maxi+tau_delta, tau_delta)
        y = z[0, spec_index, :, pick[1]]
        spec_color = spec_colors[spec_index]
        spec_label = spec_labels[spec_index]
        axe[row, col].plot(x, y, color = spec_color, linestyle = '-', linewidth = line_width_alp, label = spec_label, alpha = 0.25)
        axe[row, col].set_xlabel(xlabel = 'Time', fontsize = font_size_alp)
        axe[row, col].set_ylabel(ylabel = 'Molecule Count', fontsize = font_size_alp)
        axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
        axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
        axe[row, col].set_xlim(x_limes)
        axe[row, col].set_ylim(y_limes)
        axe[row, col].set_title(label = 'Cell Scale'.upper(), fontsize = font_size_alp)
        if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
        axe[row, col].text(x = 0.5*time_maxi, y = 0.5*y_limes[1], s = 'Cell-Cell\nSignaling'.upper()+'\n'+['OFF', 'ON'][rule_index], color = ['red', 'darkturquoise'][rule_index], size = font_size_chi, weight = 'bold', ha = 'center', va = 'center', bbox = text_box)
# plt.show()

# SBI [Colander]

specs = ['NT', 'G', 'FT']
spec_labels = ['NANOG', 'GATA6', 'FGF4']
spec_colors = ['tab:blue', 'tab:orange', 'tab:green']
line_width_alp = 5 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+2, axe_zero[1]+2+cols)
axe = axe_mat[row_plan, col_plan]
_rows = [min(range(rows)), max(range(rows))]
_cols = [min(range(cols)), max(range(cols))]
x_limes = (time_mini, time_maxi)
y_limes = (-50, 1050)
x_ticks = np.array([_ for _ in x if _ % 4 == 0]).astype('int') if verbose else []
x_tick_labels = [_ if _ % 8 == 0 else None for _ in x_ticks] if verbose else []
y_ticks = np.linspace(y_limes[0], y_limes[1], int((y_limes[1]-y_limes[0])/50)+1).astype('int') if verbose else []
y_tick_labels = [_ if _ % 250 == 0 else None for _ in y_ticks] if verbose else []
_tit = 'B'
tits = np.array([[f'[{_tit+str(rule_index+1)}]'] for rule_index in range(len(rules))])
legend_locks = ['center right', 'upper right']
for rule in rules:
    rule_index = rules.index(rule)
    row = _rows[rule_index]
    col = _cols[rule_index]
    tit = tits[rule_index, col]
    legend_lock = legend_locks[rule_index]
    z = _colander[rule_index]
    for spec in specs:
        spec_index = specs.index(spec)
        # x = np.arange(0, tau_maxi+tau_delta, tau_delta)
        y = z[spec][0, :, pick[1]]
        spec_color = spec_colors[spec_index]
        spec_label = spec_labels[spec_index]
        axe[row, col].plot(x, y, color = spec_color, linestyle = '-', linewidth = line_width_alp, label = spec_label, alpha = 1)
        axe[row, col].set_xlabel(xlabel = 'Time', fontsize = font_size_alp)
        axe[row, col].set_ylabel(ylabel = 'Molecule Count', fontsize = font_size_alp)
        axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
        axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
        axe[row, col].set_xlim(x_limes)
        axe[row, col].set_ylim(y_limes)
        axe[row, col].set_title(label = 'Cell Scale'.upper(), fontsize = font_size_alp)
        if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
        axe[row, col].legend(loc = legend_lock, fontsize = font_size_bet, ncols = 1, title = 'Total', title_fontsize = font_size_bet, borderaxespad = 0.25)
# plt.show()

# SBI [Classifier]

fates = ['NT', 'G', 'U']
fate_labels = ['EPI', 'PRE', 'UND']
fate_colors = ['tab:blue', 'tab:orange', 'tab:olive']
line_width_alp = 1 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+3, axe_zero[1]+3+cols)
axe = axe_mat[row_plan, col_plan]
_rows = [min(range(rows)), max(range(rows))]
_cols = [min(range(cols)), max(range(cols))]
_tit = 'C'
tits = np.array([[f'[{_tit+str(rule_index+1)}]'] for rule_index in range(len(rules))])
color_map = 'tab10'
tableau_colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
i = 0 # simulation_index
j = int(48/tau_delta) # tau_index
_heat_map = {rule_index: {fate: tableau_colors.index(fate_colors[fates.index(fate)])*_classifier[rule_index][fate][i, j, :].reshape(cellulate) for fate in fates} for rule_index in range(len(rules))}
for rule in rules:
    rule_index = rules.index(rule)
    row = _rows[rule_index]
    col = _cols[rule_index]
    tit = tits[rule_index, col]
    for fate in fates:
        fate_index = fates.index(fate)
        w = _heat_map[rule_index][fate]
        if fate_index == 0:
            heat_map = w
        else:
            heat_map += w
        axe[row, col].imshow(X = heat_map, cmap = color_map, vmin = 0, vmax = 10, origin = 'upper')
        if not verbose: axe[row, col].set_xlabel(xlabel = 'Cell Position', fontsize = font_size_alp)
        if not verbose: axe[row, col].set_ylabel(ylabel = 'Cell Position', fontsize = font_size_alp)
        ticks_major = np.arange(0, cellulate[0], 1) # cellulate[1]
        ticks_minor = np.array([_ for _ in np.arange(0, cellulate[1], 0.5) if _ % 1 == 0.5]) # cellulate[0]
        tick_labels_major = [None]*len(ticks_major)
        tick_labels_minor = [None]*len(ticks_minor)
        axe[row, col].set_xticks(ticks = ticks_major, labels = tick_labels_major, fontsize = font_size_bet, minor = False)
        axe[row, col].set_yticks(ticks = ticks_major, labels = tick_labels_major, fontsize = font_size_bet, minor = False)
        axe[row, col].set_xticks(ticks = ticks_minor, labels = tick_labels_minor, fontsize = font_size_bet, minor = True)
        axe[row, col].set_yticks(ticks = ticks_minor, labels = tick_labels_minor, fontsize = font_size_bet, minor = True)
        axe[row, col].tick_params(which = 'both', bottom = False, left = False)
        axe[row, col].spines[:].set_visible(True)
        axe[row, col].spines[:].set_color('w')
        axe[row, col].spines[:].set_linewidth(line_width_alp)
        axe[row, col].set_title(label = 'Tissue Scale'.upper(), fontsize = font_size_alp)
        if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
        axe[row, col].grid(which = 'minor', color = 'w', linewidth = line_width_alp, alpha = 1)
# plt.show()

# SBI [Counter]

fates = ['NT', 'G', 'U']
fate_labels = ['EPI', 'PRE', 'UND']
fate_colors = ['tab:blue', 'tab:orange', 'tab:olive']
line_width_alp = 5 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
line_width_bet = 3 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+4, axe_zero[1]+4+cols)
axe = axe_mat[row_plan, col_plan]
_rows = [min(range(rows)), max(range(rows))]
_cols = [min(range(cols)), max(range(cols))]
x_limes = (time_mini, time_maxi)
y_limes = (-5, 105)
x_ticks = np.array([_ for _ in x if _ % 4 == 0]).astype('int') if verbose else []
x_tick_labels = [_ if _ % 8 == 0 else None for _ in x_ticks] if verbose else []
y_ticks = np.linspace(y_limes[0], y_limes[1], int((y_limes[1]-y_limes[0])/5)+1).astype('int') if verbose else []
y_tick_labels = [_ if _ % 25 == 0 else None for _ in y_ticks] if verbose else []
_tit = 'D'
tits = np.array([[f'[{_tit+str(rule_index+1)}]'] for rule_index in range(len(rules))])
fate_targets = np.array([[100, 0, 0], [40, 60, 0]]).astype('int')
fate_target_labels = [f'{fate_label} Target' for fate_label in fate_labels] if verbose else [None, None, None]
fate_target_colors = ['b', 'r', 'y']
legend_locks = ['center right', 'upper right']
for rule in rules:
    rule_index = rules.index(rule)
    row = _rows[rule_index]
    col = _cols[rule_index]
    tit = tits[rule_index, col]
    legend_lock = legend_locks[rule_index]
    z = _counter[rule_index]
    for fate in fates:
        fate_index = fates.index(fate)
        # x = np.arange(0, tau_maxi+tau_delta, tau_delta)
        y = z[fate][0, :]
        fate_color = fate_colors[fate_index]
        fate_label = fate_labels[fate_index]
        fate_target = fate_targets[rule_index, fate_index]
        fate_target_color = fate_target_colors[fate_index]
        fate_target_label = fate_target_labels[fate_index]
        axe[row, col].plot(x, y, color = fate_color, linestyle = '-', linewidth = line_width_alp, label = fate_label, alpha = 0.75)
        axe[row, col].axhline(y = fate_target, xmin = time_mini, xmax = time_maxi, color = fate_target_color, linestyle = '--', linewidth = line_width_bet, alpha = 0.5, label = fate_target_label)
        axe[row, col].set_xlabel(xlabel = 'Time', fontsize = font_size_alp)
        axe[row, col].set_ylabel(ylabel = 'Cell Count', fontsize = font_size_alp)
        axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
        axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
        axe[row, col].set_xlim(x_limes)
        axe[row, col].set_ylim(y_limes)
        axe[row, col].set_title(label = 'Tissue Scale'.upper(), fontsize = font_size_alp)
        if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
        axe[row, col].legend(loc = legend_lock, fontsize = font_size_bet, ncols = 1)
# plt.show()

# SBI [Scorer]

line_width_alp = 5 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
line_width_bet = 2 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+5, axe_zero[1]+5+cols)
axe = axe_mat[row_plan, col_plan]
_rows = [min(range(rows)), max(range(rows))]
_cols = [min(range(cols)), max(range(cols))]
x_limes = (time_mini, time_maxi)
y_limes = (0, 1)
x_ticks = np.array([_ for _ in x if _ % 4 == 0]).astype('int') if verbose else []
x_tick_labels = [_ if _ % 8 == 0 else None for _ in x_ticks] if verbose else []
y_ticks = np.linspace(y_limes[0], y_limes[1], int((y_limes[1]-y_limes[0])/0.05)+1) if verbose else []
y_tick_labels = [_ if _ % 0.25 == 0 else None for _ in np.round(y_ticks, 2)] if verbose else []
_tit = 'E'
tits = np.array([[f'[{_tit+str(rule_index+1)}]'] for rule_index in range(len(rules))])
text_box = {'boxstyle': 'round', 'linestyle': '-.', 'linewidth': line_width_bet, 'pad': 0.25, 'facecolor': 'white', 'edgecolor': 'tab:gray', 'alpha': 0.75}
score_labels = [None]*rows
score_colors = ['darkcyan', 'darkmagenta', 'darkcyan']
for rule in rules:
    rule_index = rules.index(rule)
    row = _rows[rule_index]
    col = _cols[rule_index]
    tit = tits[rule_index, col]
    z = _scorer[rule_index]
    # x = np.arange(0, tau_maxi+tau_delta, tau_delta)
    y = z[0, 0, :, 0]
    score_color = score_colors[row]
    score_label = score_labels[row]
    axe[row, col].plot(x, y, color = score_color, linestyle = '-', linewidth = line_width_alp, label = score_label, alpha = 1)
    axe[row, col].set_xlabel(xlabel = 'Time', fontsize = font_size_alp)
    axe[row, col].set_ylabel(ylabel = 'Pattern Score', fontsize = font_size_alp)
    axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
    axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
    axe[row, col].set_xlim(x_limes)
    axe[row, col].set_ylim(y_limes)
    axe[row, col].set_title(label = 'Tissue Scale'.upper(), fontsize = font_size_alp)
    if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    axe[row, col].text(x = 0.5*time_maxi, y = 0.125*y_limes[1], s = f'Marginal\nPattern Score {rule_index+1}', color = 'tab:gray', size = font_size_chi, ha = 'center', va = 'center', bbox = text_box)
row = 1
col = 0
tit = f'[{_tit}]'
z = np.array(list(_scorer.values()))
# x = np.arange(0, tau_maxi+tau_delta, tau_delta)
y = np.power(np.prod(z, 0), 1/len(rules)).flatten()
_z = (32, 48)
_x = np.array([_ if _ >= _z[0] or _z[1] <= _ else np.nan for _ in x])
_y = np.array([np.nan if np.isnan(_x[_]) else y[_] for _ in range(y.size)])
score_color = score_colors[row]
score_label = score_labels[row]
axe[row, col].plot(x, y, color = score_color, linestyle = '-', linewidth = line_width_alp, label = None, alpha = 1)
axe[row, col].fill_between(x = _x, y1 = _y, y2 = 0, color = score_color, alpha = 0.05)
axe[row, col].set_xlabel(xlabel = 'Time', fontsize = font_size_alp)
axe[row, col].set_ylabel(ylabel = 'Pattern Score', fontsize = font_size_alp)
axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
axe[row, col].set_xlim(x_limes)
axe[row, col].set_ylim(y_limes)
axe[row, col].set_title(label = 'Tissue Scale'.upper(), fontsize = font_size_alp)
if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
axe[row, col].text(x = 0.5*time_maxi, y = 0.175*y_limes[1], s = 'Joint\nPattern\nScore', color = 'tab:gray', size = font_size_chi, ha = 'center', va = 'center', bbox = text_box)
# plt.show()

# SBI [Synthetic Scorer]

line_width_alp = 7 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
line_width_bet = 3 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+6, axe_zero[1]+6+cols)
axe = axe_mat[row_plan, col_plan]
x_limes = (time_mini, time_maxi)
y_limes = (-0.05, 1.05)
x_ticks = np.array([_ for _ in x if _ % 4 == 0]).astype('int') if verbose else []
x_tick_labels = [_ if _ % 8 == 0 else None for _ in x_ticks] if verbose else []
y_ticks = np.linspace(y_limes[0], y_limes[1], int((y_limes[1]-y_limes[0])/0.05)+1) if verbose else []
y_tick_labels = [_ if _ % 0.25 == 0 else None for _ in np.round(y_ticks, 2)] if verbose else []
text_box = {'boxstyle': 'round', 'linestyle': '-.', 'linewidth': line_width_bet, 'pad': 0.25, 'facecolor': 'white', 'edgecolor': 'tab:gray', 'alpha': 0.75}
score_labels = [None]
score_colors = ['magenta']
tit = '[F1]'
row = 0
col = 0
_x = (32, 48)
_y = (0, 1)
# x = np.arange(0, tau_maxi+tau_delta, tau_delta)
y = np.array([1 if _ >= _x[0] or _x[1] <= _ else np.nan for _ in x])
score_color = score_colors[row]
score_label = score_labels[row]
axe[row, col].plot(x, y, color = score_color, linestyle = '-', linewidth = line_width_alp, label = None, alpha = 1)
axe[row, col].fill_between(x = _x, y1 = _y[0], y2 = _y[1], color = score_color, alpha = 0.05)
axe[row, col].set_xlabel(xlabel = 'Time', fontsize = font_size_alp)
axe[row, col].set_ylabel(ylabel = 'Synthetic Score', fontsize = font_size_alp)
axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
axe[row, col].set_xlim(x_limes)
axe[row, col].set_ylim(y_limes)
axe[row, col].set_title(label = 'Tissue Scale'.upper(), fontsize = font_size_alp)
if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
axe[row, col].text(x = 0.225*time_maxi, y = 0.9*(y_limes[1]+y_limes[0]), s = 'Synthetic\nScore', color = 'tab:gray', size = font_size_chi, ha = 'center', va = 'center', bbox = text_box)
# plt.show()

# SBI [Meta Scorer]

line_width_alp = 7 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
line_width_bet = 3 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+7, axe_zero[1]+7+cols)
axe = axe_mat[row_plan, col_plan]
text_box = {'boxstyle': 'round', 'linestyle': '-.', 'linewidth': line_width_bet, 'pad': 0.25, 'facecolor': 'white', 'edgecolor': 'tab:gray', 'alpha': 0.75}
tit = '[G1]'
row = 0
col = 0
# x = np.arange(0, tau_maxi+tau_delta, tau_delta)
y = [1, 0.375, 0.75, 0.5, 0.125]
z = list(range(len(y)))
x_limes = (min(z)-0.625, max(z)+0.625)
y_limes = (-0.05, 1.05)
x_ticks = z if verbose else []
x_tick_labels = z if verbose else []
y_ticks = np.linspace(y_limes[0], y_limes[1], int((y_limes[1]-y_limes[0])/0.05)+1) if verbose else []
y_tick_labels = [_ if _ % 0.25 == 0 else None for _ in np.round(y_ticks, 2)] if verbose else []
score_labels = [None]*len(z)
score_colors = ['magenta', 'lightgray', 'lightgray', 'lightgray', 'lightgray'] # ['hotpink', 'lightgreen', 'darkturquoise', 'darkviolet', 'lightcoral']
axe[row, col].bar(x = z, height = y, width = 0.75, color = score_colors, edgecolor = 'tab:gray', linewidth = line_width_bet, label = score_labels, alpha = 1)
axe[row, col].set_xlabel(xlabel = 'Trial', fontsize = font_size_alp)
axe[row, col].set_ylabel(ylabel = 'Meta Score', fontsize = font_size_alp)
axe[row, col].set_xticks(ticks = x_ticks, labels = x_tick_labels, fontsize = font_size_bet)
axe[row, col].set_yticks(ticks = y_ticks, labels = y_tick_labels, fontsize = font_size_bet)
axe[row, col].set_xlim(x_limes)
axe[row, col].set_ylim(y_limes)
axe[row, col].set_title(label = 'Meta-Analysis'.upper(), fontsize = font_size_alp)
if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
axe[row, col].text(x = (1-0.0375)*(x_limes[1]+x_limes[0]), y = 0.9*(y_limes[1]+y_limes[0]), s = 'Meta\nScore', color = 'tab:gray', size = font_size_chi, ha = 'center', va = 'center', bbox = text_box)
# plt.show()

# SBI [ANN | Posterior | Mixer | Prior]

rows = 3
cols = 3
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+6, axe_zero[1]+6+cols)
axe = axe_mat[row_plan, col_plan]
text_box_colors = ['teal', 'azure'][::-1] # {['skyblue', 'midnightblue'], ['steelblue', 'aliceblue'], ['teal', 'azure']}
text_box_styles = ['square', 'rarrow', 'larrow', 'roundtooth', 'circle']
text_box_line_widths = [7, 5, 5, 5, 7] # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
text_box_pads = [0.75, 0.5, 0.5, 0.5, 0.75]
text_box_verses = ['ANN\nSNPE', 'Posterior', 'Prior', 'Mixture', 'Mixer'] # {'ANN', 'SNPE'}
text_box_font_sizes = [47, 43, 43, 43, 47] # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
text_box_pots = [[1, 0], [1, 1], [1, 2], [2, 1], [2, 2]]
tits = ['[F]', '[G]', '[H]', '[G2]', '[H2]']
for text_box_pot in text_box_pots:
    text_box_index = text_box_pots.index(text_box_pot)
    row, col = text_box_pot
    text_box_style = text_box_styles[text_box_index]
    text_box_line_width = text_box_line_widths[text_box_index]
    text_box_pad = text_box_pads[text_box_index]
    text_box = {'boxstyle': text_box_style, 'linestyle': '-.', 'linewidth': text_box_line_width, 'pad': text_box_pad, 'facecolor': text_box_colors[0], 'edgecolor': text_box_colors[1], 'alpha': 1}
    text_box_verse = text_box_verses[text_box_index]
    text_box_font_size = text_box_font_sizes[text_box_index]
    axe[row, col].text(x = 0.5, y = 0.5, s = text_box_verse, color = text_box_colors[1], size = text_box_font_size, weight = 'bold', ha = 'center', va = 'center', bbox = text_box)
    tit = tits[text_box_index]
    if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    axe[row, col].axis('off')
# plt.show()

# SBI [Simulator]

rows = 3
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
row_plan = slice(axe_zero[0]-1, axe_zero[0]-1+rows)
col_plan = slice(axe_zero[1]+0, axe_zero[1]+0+cols)
axe = axe_mat[row_plan, col_plan]
row = 1
col = 0
text_box_colors = ['teal', 'azure'][::-1] # {['skyblue', 'midnightblue'], ['steelblue', 'aliceblue'], ['teal', 'azure']}
text_box = {'boxstyle': 'round4', 'linestyle': '-.', 'linewidth': 7, 'pad': 0.75, 'facecolor': text_box_colors[0], 'edgecolor': text_box_colors[1], 'alpha': 1}
axe[row, col].text(x = 0.5, y = 0.5, s = 'Simulator', color = text_box_colors[1], size = 47, weight = 'bold', ha = 'center', va = 'center', bbox = text_box)
tit = '[A]'
if verbose: axe[row, col].set_title(label = tit, loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
axe[row, col].axis('off')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig1'
fig_forts = ['tiff', 'svg', 'eps', 'pdf', 'png'] # {TIFF, SVG, EPS, PDF, PNG}
fig_keros = [{'compression': 'tiff_lzw'}, None, None, None, None]
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
