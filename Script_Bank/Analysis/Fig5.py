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
fig_size_base = 7.5 # {2.5, 3.75, 5, 7.5, 10}

#%%# Retrieve Mapes

from Utilities import posterior_appraisal_selection

def retrieve_mapes(data_paths, acts, observers, curbs, verbose = False):
    mapes = list()
    for index in range(len(data_paths)):
        data_path = data_paths[index]
        act = acts[index]
        observe = observers[index]
        curb = curbs[index]
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
        _tag = f'Act_{act}_Observe_{observe}_{curb.capitalize()}'
        postage = None # postage = '_Posterior.pkl'
        posterior = posterior_appraisal_selection(path, _tag, postage, verbose = True)
        _mape = posterior.map().numpy()
        mape = [int(round(_mape[index], 0)) if _mape[index] > 1 else round(_mape[index], 2) for index in range(_mape.size)]
        if verbose: print(f'MAPE!\n{data_path}\n{_tag}\n\t{mape}')
        mapes.append(mape)
    return mapes

#%%# Data Preparation

from Fig0_Services import retrieve_para_set_truths
from Fig0_Services import objective_fun_counter

data_paths = ['Shallow_Grid_1_N_Link', 'Shallow_Grid_1_Rule_Kern'] # data_path
nooks = [0, 0] # nook
acts = [7, 3] # act
observers = [1, 1] # observe
curbs = ['Mid', 'Mid'] # curb
mapes = retrieve_mapes(data_paths, acts, observers, curbs, verbose = True) # mape
para_key_sets = [ # para_key_set
    ['N_N', 'G_G', 'FC_N', 'G_EA', 'G_N', 'N_G', 'FC_G', 'N_EA', 'MRNA', 'PRO', 'td_FC', 'td_FM', 'tau_C', 'tau_M', 'tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP', 'chi_auto'],
    ['N_N', 'G_G', 'G_N', 'N_G']
]
para_value_sets = [ # para_value_set
    [(0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 1000), (0, 250), (0, 1000), (300, 28800), (300, 28800), (30, 5*900), (30, 2*2100), (10*30, 43200), (30, 43200), (10*30, 43200), (30, 43200), (0, 1)],
    [(0, 1000), (0, 1000), (0, 1000), (0, 1000)]
]
para_set_modes = [0, 0] # para_set_mode
para_set_truths = retrieve_para_set_truths(mapes, para_key_sets, para_value_sets, para_set_modes, verbose = True) # para_set_true

keys = ['data_path', 'nook', 'act', 'observe', 'curb', 'mape', 'para_set_true']
values = [data_paths, nooks, acts, observers, curbs, mapes, para_set_truths]
memos = {data_paths[_]: {key: value[_] for key, value in zip(keys, values)} for _ in range(len(data_paths))}

species = ['N', 'G', 'NP'] # ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA']
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.25
time_slit = 1
score_mini = 0
score_maxi = 1
score_slit = 0.05

cellulates = [(1, 5), (2, 5), (3, 5), (5, 5), (7, 5), (5, 10), (5, 13), (5, 15), (5, 17), (10, 10), (10, 15), (15, 15), (20, 20)] # (cell_layers, layer_cells) # [(1, 5), (2, 5), (3, 5), (5, 5), (7, 5), (5, 10), (5, 13), (5, 15), (5, 17), (10, 10), (10, 15), (15, 15), (20, 20)]

#%%# Retrieve Data [Cellulate /\ Memo]

def retrieve_data_cellulate(memo, cellulate, reiteration, lot, verbose = False):
    from Utilities import simul_data_load
    if verbose: print(f"{' '*8}{cellulate}")
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    if lot == 0:
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
    elif lot == 1:
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
    tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate = (theta_set, trajectory_set)
    return data_cellulate

def retrieve_data_memo(memo, cellulates, rules, lot, verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    data_memo = {cellulate: [None]*len(rules) for cellulate in cellulates}
    nook = memo['nook']
    for cellulate in cellulates:
        reiterations = [f'Reiterate_Nook_{cellulate[0]}_{cellulate[1]}', f'Reiterate_{cellulate[0]}_{cellulate[1]}']
        if nook:
            for rule in rules:
                rule_index = rules.index(rule)
                reiteration = reiterations[rule_index]
                data_cellulate = retrieve_data_cellulate(memo, cellulate, reiteration, verbose)
                data_memo[cellulate][rule_index] = data_cellulate[1]
        else:
            reiteration = reiterations[1]
            data_cellulate = retrieve_data_cellulate(memo, cellulate, reiteration, lot, verbose)
            splitter = int(data_cellulate[1].shape[1]/len(rules))
            rule_split = [(rules.index(rule)*splitter, (rules.index(rule)+1)*splitter) for rule in rules]
            for rule in rules:
                rule_index = rules.index(rule)
                split = rule_split[rule_index]
                data_memo[cellulate][rule_index] = data_cellulate[1][:, split[0]:split[1]]
    return data_memo

#%%# Create Figure!

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

rows = 2
cols = 2
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = True)

lots = [0, 1]

for lot in lots:
    
    # lot = 0 # {0, 1}
    pats = ['ITWT', 'RTM'] # {'ITWT': 'Inferred-Theoretical Wild-Type', 'RTM': 'Reinferred-Theoretical Mutant'}
    
    if lot == 0:
        rug = 1
        rules = [0, 1]
        aim_NT = [1, 2/5]
        aim_G = [1-_ for _ in aim_NT]
    elif lot == 1:
        rug = 0
        rules = [1]
        aim_NT = [2/5]
        aim_G = [1-_ for _ in aim_NT]

    # Visualize Data Memo [Counter | Ratio | Cellulate]
    
    cellulates = [(1, 5), (2, 5), (3, 5), (5, 5), (7, 5), (5, 10), (5, 13), (5, 15), (5, 17), (10, 10), (10, 15), (15, 15), (20, 20)]
    species = ['N', 'G', 'NP']
    
    _memo = data_paths[lot]
    memo = memos[_memo]
    data_memo = retrieve_data_memo(memo, cellulates, rules, lot, verbose = True)
    
    synopses = {('Sand', 'Mean'): 1, ('Interquantile_Range', 'Quantile'): 0}
    interquantile_range = [0.25, 0.75] # {[0.25, 0.75], [0.025, 0.975]}
    quantile = [0.5]
    tau_mini = 48
    tau_maxi = 48
    cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
    cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
    cocks = ['tab:blue', 'tab:orange', 'tab:olive'] # list(matplotlib.colors.BASE_COLORS.keys())
    edges = [':', '-', '--']
    clues = ['NT', 'G', 'U']
    clue_labels = ['EPI', 'PRE', 'UND']
    rows = 1
    cols = 1
    # fig_size = (cols*fig_size_base, rows*fig_size_base)
    # fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
    rats = np.zeros((len(rules), len(clues), len(cellulates)))
    ratios = np.zeros((len(rules), len(clues), len(cellulates)))
    tits = ['[A]', '[C]']
    if lot == 0:
        axe = axe_mat[0:1, 0:1]
    elif lot == 1:
        axe = axe_mat[1:2, 0:1]
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        for rule in rules:
            rule_index = rules.index(rule)
            trajectory_set = data_memo[cellulate][rule_index]
            objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
            counter = objective_fun_counter(objective_fun, tau_mini = tau_mini, tau_maxi = tau_maxi)
            x = objective_fun.tau
            y = counter
            for clue in clues:
                clue_index = clues.index(clue)
                if synopses[('Sand', 'Mean')]:
                    y_mean = np.mean(y[clue], 0).flatten() if clue != 'U' else np.mean(cellulate_cells-(y['NT']+y['G']), 0).flatten()
                    y_sand = np.std(y[clue], 0).flatten() if clue != 'U' else np.std(cellulate_cells-(y['NT']+y['G']), 0).flatten()
                    y_ratio = y_sand/y_mean
                    rats[rule_index, clue_index, cellulate_index] = (cellulate_cells-y_mean[-1])/np.sqrt(cellulate_cells*y_mean[-1]*(cellulate_cells-y_mean[-1])) # 1/np.sqrt(y_mean[-1])
                    ratios[rule_index, clue_index, cellulate_index] = np.mean(y_ratio)
                elif synopses[('Interquantile_Range', 'Quantile')]:
                    pass
    
    font_size_alp = 29 # {11, 13, 23, 29}
    font_size_bet = 23 # {11, 13, 23, 29}
    font_size_chi = 13 # {11, 13, 23, 29}
    font_size_ate = 17 # Annotate!
    font_size_bat = 19 # Annotate!
    _x = cells
    x_limes = (2.5, 750)
    row = 0
    col = 0
    _axe = axe[row, col].inset_axes([0, 0, 0.375, 0.375])
    _axe.axhline(y = 1, xmin = 0, xmax = 1000, color = 'm', linewidth = 1.5, alpha = 0.5, zorder = 0)
    axe[row, col].axhline(y = 10, xmin = 0, xmax = 1000, color = 'tab:gray', linestyle = edges[0], linewidth = 2, marker = '.', fillstyle = 'none', label = r'$CV_0$', alpha = 0.75, zorder = 0)
    axe[row, col].axhline(y = 10, xmin = 0, xmax = 1000, color = 'tab:gray', linestyle = edges[1], linewidth = 2, marker = '.', fillstyle = 'full', label = r'$CV_1$', alpha = 0.75, zorder = 0)
    axe[row, col].axhline(y = 10, xmin = 0, xmax = 1000, color = 'tab:gray', linestyle = edges[2], linewidth = 1.5, marker = '.', fillstyle = 'full', label = r'$CV_0 \slash CV_1$', alpha = 0.75, zorder = 0)
    for clue in clues:
        clue_index = clues.index(clue)
        axe[row, col].plot(cells, rats[rug, clue_index], color = cocks[clue_index], linestyle = edges[0], linewidth = 2, marker = '.', fillstyle = 'none', label = None)
        axe[row, col].plot(cells, ratios[rug, clue_index], color = cocks[clue_index], linestyle = edges[1], linewidth = 2, marker = '.', fillstyle = 'full', label = None)
        _y_mini = 0.025
        _y_maxi = 1.25 # {1, 1.25}
        _y_numb = int(_y_maxi/_y_mini)
        _y = np.round(np.linspace(_y_mini, _y_maxi, _y_numb), 7)
        y_limes = (_y_mini, _y_maxi)
        axe[row, col].set_title(label = pats[lot] + ' ' + r'(CV $\equiv$ Coefficient of Variation)', fontsize = font_size_bet)
        axe[row, col].set_title(label = tits[lot], loc = 'left', pad = 17, fontsize = font_size_alp, fontweight = 'bold', position = (-0.1875, 1))
        axe[row, col].set_xscale(value = 'log')
        axe[row, col].set_yscale(value = 'log')
        axe[row, col].tick_params(axis = 'x', labelsize = font_size_bet, rotation = 0) # axe[row, col].set_xticks(ticks = _x, labels = [], fontsize = font_size_bet, rotation = 0)
        axe[row, col].tick_params(axis = 'y', labelsize = font_size_bet, rotation = 0) # axe[row, col].set_yticks(ticks = _y, labels = [], fontsize = font_size_bet, rotation = 0)
        axe[row, col].set_xlabel(xlabel = 'System Size [Cell Count]', fontsize = font_size_bet)
        # axe[row, col].set_ylabel(ylabel = r'$CV_0 = \frac{\eta-\mu}{\sqrt{\eta\mu(\eta-\mu)}}$     $CV_1 = \frac{\sigma}{\mu}$', fontsize = font_size_bet)
        axe[row, col].set_ylabel(ylabel = r'$CV_0$     $CV_1$', fontsize = font_size_bet)
        axe[row, col].set_xlim(x_limes)
        axe[row, col].set_ylim(y_limes)
        axe[row, col].grid(alpha = 0.375, which = 'major')
        axe[row, col].legend(fontsize = font_size_ate, loc = 'upper right' if lot == 0 else 'upper center')
        dart = {'width': 1, 'headwidth': 5, 'headlength': 3, 'shrink': 0.1, 'edgecolor': 'darkolivegreen', 'facecolor': 'w'}
        # dart_where = (100, ratios[rug, 0, cells.index(100)])
        # axe[row, col].annotate(text = r'$\eta = 100$ Cells', xy = dart_where, xytext = (dart_where[0]-25, dart_where[1]+0.125), arrowprops = dart, color = 'darkolivegreen', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'center', verticalalignment = 'center')
        dart_where = (10, rats[rug, 0, cells.index(10)])
        axe[row, col].annotate(text = r'$CV_0 = \frac{\eta-\mu}{\sqrt{\eta\mu(\eta-\mu)}}$', xy = dart_where, xytext = (dart_where[0]-2.5, dart_where[1]+0.25), arrowprops = dart, color = 'darkolivegreen', fontsize = font_size_bat, fontweight = 'book', horizontalalignment = 'center', verticalalignment = 'center')
        dart_where = (100, ratios[rug, 0, cells.index(100)])
        axe[row, col].annotate(text = r'$CV_1 = \frac{\sigma}{\mu}$', xy = dart_where, xytext = (dart_where[0]-12.5, dart_where[1]+0.125), arrowprops = dart, color = 'darkolivegreen', fontsize = font_size_bat, fontweight = 'book', horizontalalignment = 'center', verticalalignment = 'center')
        dart_where = (5, ratios[rug, 1, cells.index(5)])
        axe[row, col].annotate(text = r'$\eta = 5$ Cells', xy = dart_where, xytext = (dart_where[0]-0.25, dart_where[1]-0.125), arrowprops = dart, color = 'darkolivegreen', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'center', verticalalignment = 'center')
        dart_where = (400, ratios[rug, 2, cells.index(400)])
        axe[row, col].annotate(text = r'$\eta = 400$ Cells', xy = dart_where, xytext = (dart_where[0]-50, dart_where[1]-0.125 if lot == 0 else dart_where[1]-0.25), arrowprops = dart, color = 'darkolivegreen', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'center', verticalalignment = 'center')
        _axe.plot(cells, rats[rug, clue_index]/ratios[rug, clue_index], color = cocks[clue_index], linestyle = edges[2], linewidth = 1.5, marker = '.', label = clue_labels[clue_index])
        _y_mini = 0.875 # 0.75
        _y_maxi = 1.375 # 1.25
        _y_numb = int((_y_maxi-_y_mini)/(1-_y_mini))+1 # {4, 9}
        _y = np.round(np.linspace(_y_mini, _y_maxi, _y_numb), 7)
        y_limes = (_y_mini, _y_maxi)
        # _axe.set_title(label = '', fontsize = font_size_ate)
        _axe.set_xscale(value = 'log')
        # _axe.set_yscale(value = 'log', subs = [_ for _ in range(0, 10) if _ >= 2]) # [_ for _ in range(0, 10) if _ % 2 == 0 and _ >= 2]
        _axe.tick_params(axis = 'x', labelsize = font_size_ate, rotation = 0) # _axe.set_xticks(ticks = _x, labels = [], fontsize = font_size_ate, rotation = 0)
        _axe.set_yticks(ticks = _y, labels = [_ if _ not in y_limes and _ % 0.25 == 0 else None for _ in _y], fontsize = font_size_ate, rotation = 0) # _axe.tick_params(axis = 'y', labelsize = font_size_ate, rotation = 0)
        _axe.xaxis.tick_top()
        _axe.yaxis.tick_right()
        _axe.set_ylabel(ylabel = r'$CV_0 \slash CV_1$', fontsize = font_size_ate)
        _axe.yaxis.set_label_position('right')
        _axe.set_xlim(x_limes)
        _axe.set_ylim(y_limes)
        _axe.grid(alpha = 0.375, which = 'major')
        _axe.legend(fontsize = font_size_chi, frameon = True, framealpha = 0.25, loc = 'upper center', ncols = len(clues)-1)

# plt.show()

    # Visualize Data Memo [Counter]
    
    cellulates = [(5, 5), (10, 10), (20, 20)]
    species = ['N', 'G', 'NP']
    
    _memo = data_paths[lot]
    memo = memos[_memo]
    data_memo = retrieve_data_memo(memo, cellulates, rules, lot, verbose = True)
    
    synopses = {'Mean': 1, 'Sand': 1, 'Quantiles': 0}
    quantiles = [0.5] # {[], [0.5], [0.25, 0.5, 0.75], [0.025, 0.5, 0.975]}
    normalize = 1 # {0, 1} # Do we need to calculate the CV? Use: 'normalize = 1'!
    cells = [cellulate[0]*cellulate[1] for cellulate in cellulates]
    cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
    _cocos = ['teal', 'lime', 'magenta'] # list(matplotlib.colors.BASE_COLORS.keys())
    cocks = ['b', 'r'] # [cocos[0], cocos[1]]
    caps = ['full', 'none']
    clues = ['NT', 'G', 'U']
    rows = 1
    cols = 1
    # fig_size = (cols*fig_size_base, rows*fig_size_base)
    # fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
    # font_size_alp = 29 # {11, 13, 23, 29}
    # font_size_bet = 23 # {11, 13, 23, 29}
    # font_size_chi = 13 # {11, 13, 23, 29}
    # font_size_ate = 17 # Annotate!
    tits = ['[B]', '[D]']
    if lot == 0:
        axe = axe_mat[0:1, 1:2]
    elif lot == 1:
        axe = axe_mat[1:2, 1:2]
    rule = 1
    rule_index = rules.index(rule)
    row = 0
    col = 0
    for cellulate in cellulates:
        cellulate_index = cellulates.index(cellulate)
        cellulate_cells = cells[cellulate_index]
        normalization = np.power(cellulate_cells, normalize)
        trajectory_set = data_memo[cellulate][rule_index]
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        counter = objective_fun_counter(objective_fun, tau_mini = time_mini, tau_maxi = time_maxi)
        x = objective_fun.tau
        y = counter if not normalize else {key: 100*counter[key]/normalization for key in counter.keys()}
        coco = _cocos[cellulate_index]
        cap = caps[cellulate_index // len(cocos)]
        for clue in clues:
            clue_index = clues.index(clue)
            label = r'$\eta$' + ' = ' + f'{cellulate_cells} Cells' if clue_index == 0 else None
            if synopses['Mean']:
                if clue != 'U':
                    y_mean = np.mean(y[clue], 0).flatten()
                else:
                    y_mean = np.mean(100-(y['NT']+y['G']), 0).flatten()
                axe[row, col].plot(x, y_mean, color = coco, fillstyle = cap, linestyle = '-', marker = 'None', linewidth = 2, label = label, alpha = 0.75)
                if synopses['Sand']:
                    if clue != 'U':
                        y_sand = np.std(y[clue], 0).flatten()
                    else:
                        y_sand = np.std(100-(y['NT']+y['G']), 0).flatten()
                    y_net = y_mean-y_sand
                    y_pot = y_mean+y_sand
                    axe[row, col].plot(x, y_net, color = coco, linestyle = '--', linewidth = 1.5, alpha = 0.5)
                    axe[row, col].plot(x, y_pot, color = coco, linestyle = '--', linewidth = 1.5, alpha = 0.5)
            _x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
            _y = np.linspace(0, 100, 11).astype('int')
            axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_yticks(ticks = _y, labels = [f'{_}%' if _ % 20 == 0 else None for _ in _y], fontsize = font_size_bet, rotation = 0)
            axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
            axe[row, col].set_ylabel('System Size [Cell Count Percentage]', fontsize = font_size_bet)
            axe[row, col].set_xlim(time_mini, time_maxi) # x_limes
            axe[row, col].set_ylim(0, 100) # y_limes
            # axe[row, col].grid(alpha = 0.375)
        axe[row, col].set_title(label = f'{pats[lot]} (Tissue Level μ ± σ)', fontsize = font_size_bet)
        axe[row, col].set_title(label = tits[lot], loc = 'left', pad = 17, fontsize = font_size_alp, fontweight = 'bold', position = (-0.1875, 1))
    labels = ['EPI Target', 'PRE Target']
    axe[row, col].axhline(100*aim_NT[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[0], linestyle = ':', linewidth = 2, alpha = 0.75, label = labels[0])
    axe[row, col].axhline(100*aim_G[rule_index], time_mini/(time_maxi-time_mini), time_maxi/(time_maxi-time_mini), color = cocks[1], linestyle = ':', linewidth = 2, alpha = 0.75, label = labels[1])
    where_mu = (8.5, 52.5)
    where_mu_sigma = (8.5, 47.5)
    axe[row, col].axhline(52.5, 4/(time_maxi-time_mini), 8/(time_maxi-time_mini), color = 'tab:gray', linestyle = '-', linewidth = 1.5, alpha = 0.75, label = None)
    axe[row, col].axhline(47.5, 4/(time_maxi-time_mini), 8/(time_maxi-time_mini), color = 'tab:gray', linestyle = '--', linewidth = 1.5, alpha = 0.75, label = None)
    axe[row, col].annotate(text = r'$\mu$', xy = where_mu, xytext = where_mu, color = 'darkolivegreen', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
    axe[row, col].annotate(text = r'$\mu \pm \sigma$', xy = where_mu_sigma, xytext = where_mu_sigma, color = 'darkolivegreen', fontsize = font_size_ate, fontweight = 'book', horizontalalignment = 'left', verticalalignment = 'center')
    axe[row, col].legend(fontsize = font_size_chi+2, frameon = True, framealpha = 0.25, loc = 'upper right', ncols = 2)

plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig5'
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
