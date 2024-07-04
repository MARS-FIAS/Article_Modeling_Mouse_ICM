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
time_maxi = 96 # Maturity
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

_initiate_set_keys = ['FC_MRNA', 'FC'] # ['MRNA', 'PRO']
_initiate_set_values = [ # [0, 25, 50, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 150, 175, 200, 225, 250]
    [[0]*5, [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]], # FC_MRNA
    [[0, 250, 500, 750, 1000], [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]] # FC
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

def retrieve_data_memo_initiate(memo, cellulates, maturity, initiate_set, wait, verbose = False, _reiteration = 'Reiterate'):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    keys = list(initiate_set.keys())
    values = list(initiate_set.values())
    initiate_set_temp = list(zip(*values))
    data_memo_initiate = {(cellulate, initiate_temp): None for cellulate in cellulates for initiate_temp in initiate_set_temp}
    for cellulate in cellulates:
        for initiate_temp in initiate_set_temp:
            initiate = {keys[index]: initiate_temp[index] for index in range(len(keys))}
            _initiate = '_'.join(map(str, [f'{key}_{value}' for key, value in initiate.items()]))
            reiteration = _reiteration+'_'+f'{cellulate[0]}_{cellulate[1]}_Maturity_{maturity}_Initiate_{_initiate}_Wait_{wait}' # {Wild, Nook, Cast}
            data_cellulate_initiate = retrieve_data_cellulate_initiate(memo, cellulate, initiate, reiteration, verbose)
            data_memo_initiate[(cellulate, initiate_temp)] = data_cellulate_initiate[1]
    return data_memo_initiate

#%%# Create Figure!

rows = 2
cols = 4
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = True)

# [Raw Counter Pro]

_reiteration = 'Reiterate'

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_wait = [0, 4, 8, 12, 16, 24, 32, 40] # [0, 4, 8, 12, 16, 24, 32, 40]
_exam = [0, 250, 500, 750, 1000] # [0, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
_fate = ['NT', 'G', 'D']
_stat = ['mean', 'sand']
_data = np.full((len(_wait), len(_exam), len(_fate), len(_stat)), np.nan)

cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
normalize = 0
maturity = time_maxi
epoch = 1*maturity*int(1/time_delta)

for wait in _wait:
    wait_index = _wait.index(wait)
    i = 0
    _memo = data_paths[i]
    memo = memos[_memo]
    initiate_set_index = data_paths.index(_memo)
    initiate_set_temp = _initiate_set[initiate_set_index]
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, maturity, initiate_set_temp, wait, verbose = True, _reiteration = _reiteration)
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        initiate_set_keys = list(initiate_set_temp.keys())
        initiate_set_values = list(initiate_set_temp.values())
        initiate_set = list(zip(*initiate_set_values))
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            exam_index = initiate_index
            trajectory_set = data_memo_initiate[(cellulate, initiate)]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            y = counter if not normalize else {key: counter[key]/normalization for key in counter.keys()}
            y_mean_NT = np.mean(y['NT'], 0).flatten()
            y_sand_NT = np.std(y['NT'], 0).flatten()
            y_mean_G = np.mean(y['G'], 0).flatten()
            y_sand_G = np.std(y['G'], 0).flatten()
            y_mean_D = np.mean(100-(y['NT']+y['G']), 0).flatten()
            y_sand_D = np.std(100-(y['NT']+y['G']), 0).flatten()
            _data[wait_index, exam_index, 0, :] = [y_mean_NT[epoch], y_sand_NT[epoch]]
            _data[wait_index, exam_index, 1, :] = [y_mean_G[epoch], y_sand_G[epoch]]
            _data[wait_index, exam_index, 2, :] = [y_mean_D[epoch], y_sand_D[epoch]]
            print(f'{wait}\n\t{initiate}')

# Visualize Data Memo [Data]

wait_refer = 0
exam_refer = 0
wait_refer_index = _wait.index(wait_refer)
exam_refer_index = _exam.index(exam_refer)
data = np.full((len(_wait), len(_exam), len(_fate), 1), np.nan)
data_refer = _data[wait_refer, exam_refer, ...]

for wait in _wait:
    wait_index = _wait.index(wait)
    for exam in _exam:
        exam_index = _exam.index(exam)
        data[wait_index, exam_index, :, 0] = np.abs(data_refer[:, 0] - _data[wait_index, exam_index, :, 0])/data_refer[:, 1]
        print(f'{wait}\n\t{exam}\n\t{_data[wait_index, exam_index, :, :]}')
        
# Visualize Data Memo [Zero Draw Full]

x = np.array([[_]*len(_exam) for _ in _wait]).flatten()
y = np.array([_exam]*len(_wait)).flatten()
rows = 1
cols = len(_fate)
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = True, sharey = True, squeeze = False, figsize = fig_size, layout = 'constrained')
row = 0
extent = (min(_wait)-2*max(_wait)*0.05, max(_wait)+2*max(_wait)*0.05, min(_exam)-max(_exam)*0.05, max(_exam)+max(_exam)*0.05)
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
_logo = ['EPI', 'PRE', 'UND']
s_maxi = np.ceil(np.max(data[:, :, :, 0]))
axe = axe_mat[0:1, 1:4]
tits = ['[B]', '[C]', '[D]']
for col in range(cols):
    cbar = True
    fate_refer = _fate[col]
    fate_refer_index = _fate.index(fate_refer)
    s = data[:, :, fate_refer_index, 0]
    cater = s <= 1
    # s[s <= 1] = 0
    axe[row, col].contourf(x.reshape(cater.shape), y.reshape(cater.shape), cater, [0, 0.5, 1], alpha = 0.125, extent = extent, cmap = 'PiYG')
    _where = axe[row, col].contour(x.reshape(cater.shape), y.reshape(cater.shape), cater, [0.5], alpha = 1, extent = extent, linestyles = 'dashed', linewidths = 1)
    temp = axe[row, col].scatter(x, y, s = 47*s/s_maxi, c = s, cmap = 'cool', vmin = 0, vmax = s_maxi, zorder = 10) # {47, 53}
    if col == cols-1:
        _color_bar = fig.colorbar(temp, ax = axe[row, col])
        _color_bar.ax.tick_params(labelsize = font_size_bet)
        _color_bar.ax.set_ylabel(ylabel = 'Standardized Absolute Fate Deviation (SAFD)', fontsize = font_size_chi+3)
    axe[row, col].set_title(label = _logo[fate_refer_index]+' '+f'(Time = {maturity} Hours)', fontsize = font_size_bet)
    axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    axe[row, col].set_xticks(ticks = _wait, labels = [], fontsize = font_size_bet)
    axe[row, col].set_yticks(ticks = _exam, labels = [int(_/2) if _ % 250 == 0 else None for _ in _exam], fontsize = font_size_bet) if col == 0 else axe[row, col].set_yticks(ticks = _exam, labels = [], fontsize = font_size_bet)
    axe[row, col].set_xlim(min(_wait)-max(_wait)*0.025, max(_wait)+max(_wait)*0.025)
    axe[row, col].set_ylim(min(_exam)-max(_exam)*0.025, max(_exam)+max(_exam)*0.025)
    if col == 0: axe[row, col].set_ylabel('Mean Added Exogenous FGF4', fontsize = font_size_bet)
    # if col == 1: axe[row, col].set_xlabel('Perturbation Time [Hour]', fontsize = font_size_bet)
    axe[row, col].grid(color = 'white', alpha = 1)
    if len(_where.allsegs[0]) >= 1:
        lo = 450
        hi = 550
        dart_lo = {'width': 1, 'headwidth': 5, 'headlength': 3, 'shrink': 0.125, 'edgecolor': 'darkcyan', 'facecolor': 'w'}
        dart_hi = {'width': 1, 'headwidth': 5, 'headlength': 3, 'shrink': 0.125, 'edgecolor': 'darkmagenta', 'facecolor': 'w'}
        _where_lo = np.argmin(np.abs(_where.allsegs[0][0][:, 1] - lo)).flatten()
        where_lo = (_where.allsegs[0][0][:, 0][_where_lo].item(), lo)
        _where_hi = np.argmin(np.abs(_where.allsegs[0][0][:, 1] - hi)).flatten()
        where_hi = (_where.allsegs[0][0][:, 0][_where_hi].item(), hi)
        if col == cols-1:
            axe[row, col].annotate(text = 'SAFD > 1', xy = where_hi, xytext = (where_hi[0]+2.5, where_hi[1]), arrowprops = dart_hi, color = 'darkmagenta', fontsize = font_size_chi, fontweight = 'bold', horizontalalignment = 'left', verticalalignment = 'center')
            axe[row, col].annotate(text = 'SAFD < 1', xy = where_lo, xytext = (where_lo[0]-2.5, where_lo[1]), arrowprops = dart_lo, color = 'darkcyan', fontsize = font_size_chi, fontweight = 'bold', horizontalalignment = 'right', verticalalignment = 'center')
        else:
            axe[row, col].annotate(text = 'SAFD < 1', xy = where_lo, xytext = (where_lo[0]+2.5, where_lo[1]), arrowprops = dart_lo, color = 'darkcyan', fontsize = font_size_chi, fontweight = 'bold', horizontalalignment = 'left', verticalalignment = 'center')
            axe[row, col].annotate(text = 'SAFD > 1', xy = where_hi, xytext = (where_hi[0]-2.5, where_hi[1]), arrowprops = dart_hi, color = 'darkmagenta', fontsize = font_size_chi, fontweight = 'bold', horizontalalignment = 'right', verticalalignment = 'center')
# plt.show()

# Visualize Data Memo [Raw Counter]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule
i = 0
_memo = data_paths[i]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]
wait = 0
data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, maturity, initiate_set_temp, wait, verbose = True, _reiteration = _reiteration)

synopses = {'Mean': 1, 'Sand': 0, 'Quantiles': 0}
quantiles = [0.5] # {[], [0.5], [0.25, 0.5, 0.75], [0.025, 0.5, 0.975]}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
# cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
cocos = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan'] # ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
bears = ['tab:blue', 'tab:orange', 'olive']
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (3, 7))]
clues = ['NT', 'G']
rows = 1
cols = 1 # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
epoch = maturity*int(1/time_delta)
alpha = 0.75-0.25/2
axe = axe_mat[0:1, 0:1]
label_flag = 1
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
        for row in range(rows):
            col = 0
            y_mean_NT = np.mean(y['NT'], 0).flatten()
            y_sand_NT = np.std(y['NT'], 0).flatten()
            y_mean_G = np.mean(y['G'], 0).flatten()
            y_sand_G = np.std(y['G'], 0).flatten()
            y_mean_D = np.mean(100-(y['NT']+y['G']), 0).flatten()
            y_sand_D = np.std(100-(y['NT']+y['G']), 0).flatten()
            axe[row, col].set_title(label = f'ITWT (Time = {maturity} Hours)', fontsize = font_size_bet)
            axe[row, col].set_title(label = '[A]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
            axe[row, col].bar(initiate_index, y_mean_NT[epoch], 0.75, y_mean_G[epoch], color = cocos[0], alpha = alpha, label = 'EPI' if label_flag else None)
            axe[row, col].bar(initiate_index, y_mean_G[epoch], 0.75, 0, color = cocos[1], alpha = alpha, label = 'PRE' if label_flag else None)
            axe[row, col].bar(initiate_index, y_mean_D[epoch], 0.75, y_mean_NT[epoch]+y_mean_G[epoch], color = cocos[2], alpha = alpha, label = 'UND' if label_flag else None)
            label_flag = 0
            alp = axe[row, col].errorbar(initiate_index, y_mean_G[epoch], yerr = y_sand_G[epoch], fmt = 'None', ecolor = bears[1], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            bet = axe[row, col].errorbar(initiate_index, y_mean_G[epoch]+y_mean_NT[epoch], yerr = y_sand_NT[epoch], fmt = 'None', ecolor = bears[0], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            chi = axe[row, col].errorbar(initiate_index, y_mean_G[epoch]+y_mean_NT[epoch]+y_mean_D[epoch], yerr = y_sand_D[epoch], fmt = 'None', ecolor = bears[2], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            alp[1][0].set_marker('_')
            bet[1][0].set_marker('_')
            chi[1][0].set_marker('_')
            alp[1][0].set_markersize(7)
            bet[1][0].set_markersize(7)
            chi[1][0].set_markersize(7)
            _x = [_ for _ in x if _ % 1 == 0]
            _y = np.linspace(50, 100, 11).astype('int')
            tag = [int(_*mapes[i][13]/100) for _ in _initiate_set[i]['FC']] if percentage[i] else [int(_/2) for _ in _initiate_set[i]['FC']]
            axe[row, col].set_xticks(ticks = range(len(initiate_set)), labels = [], fontsize = font_size_bet, rotation = 0) # axe[row, col].set_xticks(ticks = range(len(initiate_set)), labels = tag, fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 10 == 0 else None for _ in _y], fontsize = font_size_bet)
            # if row == rows-1: axe[row, col].set_xlabel('Mean Added Exogenous FGF4\nPerturbation Time = 0 Hours', fontsize = font_size_bet)
            if col == 0: axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
            axe[row, col].set_xlim(-0.625, len(_initiate_set[0]['FC'])-0.375) # x_limes
            axe[row, col].set_ylim(50, 100) # y_limes
            # axe[row, col].axhline(aim_NT[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'b' if not normalize else cocks[0], linestyle = '--', alpha = 0.025)
            # axe[row, col].axhline(aim_G[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'r' if not normalize else cocks[0], linestyle = '--', alpha = 0.025)
axe[row, col].legend(loc = 'center right', fontsize = font_size_chi)
# plt.show()

# [Raw Counter Pro] [Nook]

_reiteration = 'Reiterate_Nook'

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_wait = [0, 4, 8, 12, 16, 24, 32, 40] # [0, 4, 8, 12, 16, 24, 32, 40]
_exam = [0, 250, 500, 750, 1000] # [0, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
_fate = ['NT', 'G', 'D']
_stat = ['mean', 'sand']
_data = np.full((len(_wait), len(_exam), len(_fate), len(_stat)), np.nan)

cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
normalize = 0
maturity = time_maxi
epoch = 1*maturity*int(1/time_delta)

for wait in _wait:
    wait_index = _wait.index(wait)
    i = 0
    _memo = data_paths[i]
    memo = memos[_memo]
    initiate_set_index = data_paths.index(_memo)
    initiate_set_temp = _initiate_set[initiate_set_index]
    data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, maturity, initiate_set_temp, wait, verbose = True, _reiteration = _reiteration)
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        initiate_set_keys = list(initiate_set_temp.keys())
        initiate_set_values = list(initiate_set_temp.values())
        initiate_set = list(zip(*initiate_set_values))
        for initiate in initiate_set:
            initiate_index = initiate_set.index(initiate)
            exam_index = initiate_index
            trajectory_set = data_memo_initiate[(cellulate, initiate)]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[1], aim_G = aim_G[1])
            counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
            x = objective_fun.tau
            y = counter if not normalize else {key: counter[key]/normalization for key in counter.keys()}
            y_mean_NT = np.mean(y['NT'], 0).flatten()
            y_sand_NT = np.std(y['NT'], 0).flatten()
            y_mean_G = np.mean(y['G'], 0).flatten()
            y_sand_G = np.std(y['G'], 0).flatten()
            y_mean_D = np.mean(100-(y['NT']+y['G']), 0).flatten()
            y_sand_D = np.std(100-(y['NT']+y['G']), 0).flatten()
            _data[wait_index, exam_index, 0, :] = [y_mean_NT[epoch], y_sand_NT[epoch]]
            _data[wait_index, exam_index, 1, :] = [y_mean_G[epoch], y_sand_G[epoch]]
            _data[wait_index, exam_index, 2, :] = [y_mean_D[epoch], y_sand_D[epoch]]
            print(f'{wait}\n\t{initiate}')

# Visualize Data Memo [Data] [Nook]

wait_refer = 0
exam_refer = 0
wait_refer_index = _wait.index(wait_refer)
exam_refer_index = _exam.index(exam_refer)
data = np.full((len(_wait), len(_exam), len(_fate), 1), np.nan)
data_refer = _data[wait_refer, exam_refer, ...]

for wait in _wait:
    wait_index = _wait.index(wait)
    for exam in _exam:
        exam_index = _exam.index(exam)
        data[wait_index, exam_index, :, 0] = np.abs(data_refer[:, 0] - _data[wait_index, exam_index, :, 0])/data_refer[:, 1]
        print(f'{wait}\n\t{exam}\n\t{_data[wait_index, exam_index, :, :]}')

# Visualize Data Memo [Zero Draw Full] [Nook]

x = np.array([[_]*len(_exam) for _ in _wait]).flatten()
y = np.array([_exam]*len(_wait)).flatten()
rows = 1
cols = len(_fate)
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = True, sharey = True, squeeze = False, figsize = fig_size, layout = 'constrained')
row = 0
extent = (min(_wait)-2*max(_wait)*0.05, max(_wait)+2*max(_wait)*0.05, min(_exam)-max(_exam)*0.05, max(_exam)+max(_exam)*0.05)
_logo = ['EPI', 'PRE', 'UND']
s_maxi = np.ceil(np.max(data[:, :, :, 0]))
axe = axe_mat[1:2, 1:4]
tits = ['[F]', '[G]', '[H]']
for col in range(cols):
    cbar = True
    fate_refer = _fate[col]
    fate_refer_index = _fate.index(fate_refer)
    s = data[:, :, fate_refer_index, 0]
    cater = s <= 1
    # s[s <= 1] = 0
    axe[row, col].contourf(x.reshape(cater.shape), y.reshape(cater.shape), cater, [0, 0.5, 1], alpha = 0.125, extent = extent, cmap = 'PiYG')
    _where = axe[row, col].contour(x.reshape(cater.shape), y.reshape(cater.shape), cater, [0.5], alpha = 1, extent = extent, linestyles = 'dashed', linewidths = 1)
    temp = axe[row, col].scatter(x, y, s = 47*s/s_maxi, c = s, cmap = 'cool', vmin = 0, vmax = s_maxi, zorder = 10) # {47, 53}
    if col == cols-1:
        _color_bar = fig.colorbar(temp, ax = axe[row, col])
        _color_bar.ax.tick_params(labelsize = font_size_bet)
        _color_bar.ax.set_ylabel(ylabel = 'Standardized Absolute Fate Deviation (SAFD)', fontsize = font_size_chi+3)
    # axe[row, col].set_title(label = _logo[fate_refer_index]+' '+f'(Time = {maturity} Hours)', fontsize = font_size_bet)
    axe[row, col].set_title(label = tits[col], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    axe[row, col].set_xticks(ticks = _wait, labels = _wait, fontsize = font_size_bet)
    axe[row, col].set_yticks(ticks = _exam, labels = [int(_/2) if _ % 250 == 0 else None for _ in _exam], fontsize = font_size_bet) if col == 0 else axe[row, col].set_yticks(ticks = _exam, labels = [], fontsize = font_size_bet)
    axe[row, col].set_xlim(min(_wait)-max(_wait)*0.025, max(_wait)+max(_wait)*0.025)
    axe[row, col].set_ylim(min(_exam)-max(_exam)*0.025, max(_exam)+max(_exam)*0.025)
    if col == 0: axe[row, col].set_ylabel('Mean Added Exogenous FGF4', fontsize = font_size_bet)
    if col >= 0: axe[row, col].set_xlabel('Perturbation Time [Hour]', fontsize = font_size_bet)
    axe[row, col].grid(color = 'white', alpha = 1)
    if len(_where.allsegs[0]) >= 1:
        lo = 500 if fate_refer in ['NT', 'G'] else 250
        hi = 500 if fate_refer in ['NT', 'G'] else 750
        dart_lo = {'width': 1, 'headwidth': 5, 'headlength': 3, 'shrink': 0.125, 'edgecolor': 'darkcyan', 'facecolor': 'w'}
        dart_hi = {'width': 1, 'headwidth': 5, 'headlength': 3, 'shrink': 0.125, 'edgecolor': 'darkmagenta', 'facecolor': 'w'}
        _where_lo = np.argwhere(_where.allsegs[0][0][:, 1] == lo).flatten()
        where_lo = (_where.allsegs[0][0][:, 0][_where_lo].item(), lo-50)
        _where_hi = np.argwhere(_where.allsegs[0][0][:, 1] == hi).flatten()
        where_hi = (_where.allsegs[0][0][:, 0][_where_hi].item(), hi+50)
        axe[row, col].annotate(text = 'SAFD < 1', xy = where_lo, xytext = (where_lo[0]+2.5, where_lo[1]), arrowprops = dart_lo, color = 'darkcyan', fontsize = font_size_chi, fontweight = 'bold', horizontalalignment = 'left', verticalalignment = 'center')
        axe[row, col].annotate(text = 'SAFD > 1', xy = where_hi, xytext = (where_hi[0]-2.5, where_hi[1]), arrowprops = dart_hi, color = 'darkmagenta', fontsize = font_size_chi, fontweight = 'bold', horizontalalignment = 'right', verticalalignment = 'center')
# plt.show()

# Visualize Data Memo [Raw Counter] [Nook]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule
i = 0
_memo = data_paths[i]
memo = memos[_memo]
initiate_set_index = data_paths.index(_memo)
initiate_set_temp = _initiate_set[initiate_set_index]
wait = 0
data_memo_initiate = retrieve_data_memo_initiate(memo, cellulates, maturity, initiate_set_temp, wait, verbose = True, _reiteration = _reiteration)

synopses = {'Mean': 1, 'Sand': 0, 'Quantiles': 0}
quantiles = [0.5] # {[], [0.5], [0.25, 0.5, 0.75], [0.025, 0.5, 0.975]}
normalize = 0 # {0, 1}
cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
slit = max(cells)/20 if normalize else 5
# cocos = list(matplotlib.colors.TABLEAU_COLORS.keys()) # list(matplotlib.colors.BASE_COLORS.keys())
cocos = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan'] # ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
bears = ['tab:blue', 'tab:orange', 'olive']
cocks = ['b', 'r'] # [cocos[0], cocos[1]]
caps = ['full', 'none']
edges = ['-', (0, (3, 7))]
clues = ['NT', 'G']
rows = 1
cols = 1 # {'NT', 'G'}
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
epoch = maturity*int(1/time_delta)
alpha = 0.75-0.25/2
axe = axe_mat[1:2, 0:1]
label_flag = 1
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
        for row in range(rows):
            col = 0
            y_mean_NT = np.mean(y['NT'], 0).flatten()
            y_sand_NT = np.std(y['NT'], 0).flatten()
            y_mean_G = np.mean(y['G'], 0).flatten()
            y_sand_G = np.std(y['G'], 0).flatten()
            y_mean_D = np.mean(100-(y['NT']+y['G']), 0).flatten()
            y_sand_D = np.std(100-(y['NT']+y['G']), 0).flatten()
            axe[row, col].set_title(label = f'TM-FGF4 (Time = {maturity} Hours)', fontsize = font_size_bet)
            axe[row, col].set_title(label = '[E]', loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
            axe[row, col].bar(initiate_index, y_mean_NT[epoch], 0.75, y_mean_G[epoch], color = cocos[0], alpha = alpha, label = 'EPI' if label_flag else None)
            axe[row, col].bar(initiate_index, y_mean_G[epoch], 0.75, 0, color = cocos[1], alpha = alpha, label = 'PRE' if label_flag else None)
            axe[row, col].bar(initiate_index, y_mean_D[epoch], 0.75, y_mean_NT[epoch]+y_mean_G[epoch], color = cocos[2], alpha = alpha, label = 'UND' if label_flag else None)
            label_flag = 0
            alp = axe[row, col].errorbar(initiate_index, y_mean_G[epoch], yerr = y_sand_G[epoch], fmt = 'None', ecolor = bears[1], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            bet = axe[row, col].errorbar(initiate_index, y_mean_G[epoch]+y_mean_NT[epoch], yerr = y_sand_NT[epoch], fmt = 'None', ecolor = bears[0], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            chi = axe[row, col].errorbar(initiate_index, y_mean_G[epoch]+y_mean_NT[epoch]+y_mean_D[epoch], yerr = y_sand_D[epoch], fmt = 'None', ecolor = bears[2], elinewidth = 1, capsize = 0, barsabove = True, uplims = True)
            alp[1][0].set_marker('_')
            bet[1][0].set_marker('_')
            chi[1][0].set_marker('_')
            alp[1][0].set_markersize(7)
            bet[1][0].set_markersize(7)
            chi[1][0].set_markersize(7)
            _x = [_ for _ in x if _ % 1 == 0]
            _y = np.linspace(0, 100, 11).astype('int')
            tag = [int(_*mapes[i][13]/100) for _ in _initiate_set[i]['FC']] if percentage[i] else [int(_/2) for _ in _initiate_set[i]['FC']]
            axe[row, col].set_xticks(ticks = range(len(initiate_set)), labels = tag, fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_yticks(ticks = _y, labels = [_ if _ % 20 == 0 else None for _ in _y], fontsize = font_size_bet)
            if row == 0: axe[row, col].set_xlabel('Mean Added Exogenous FGF4\nPerturbation Time = 0 Hours', fontsize = font_size_bet)
            if col == 0: axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
            axe[row, col].set_xlim(-0.625, len(_initiate_set[0]['FC'])-0.375) # x_limes
            axe[row, col].set_ylim(0, 100) # y_limes
            # axe[row, col].axhline(aim_NT[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'b' if not normalize else cocks[0], linestyle = '--', alpha = 0.025)
            # axe[row, col].axhline(aim_G[1]*cellulate_cells/normalization, time_mini, time_maxi, color = 'r' if not normalize else cocks[0], linestyle = '--', alpha = 0.025)
axe[row, col].legend(loc = 'center right', fontsize = font_size_chi)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig11'
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
