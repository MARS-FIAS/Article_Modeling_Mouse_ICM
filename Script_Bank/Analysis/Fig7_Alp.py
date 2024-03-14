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

from Fig0_Services import retrieve_mapes, retrieve_para_set_truths, retrieve_data_memo
from Fig0_Services import objective_fun_classifier
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

species = ['N', 'G', 'NP'] # ['N', 'G', 'N_MRNA', 'G_MRNA', 'NP', 'FC', 'FC_MRNA', 'FM', 'FD', 'EI', 'EA']
time_mini = 0
time_maxi = 48
time_unit = 'Hours'
time_delta = 0.25
time_slit = 1
score_mini = 0
score_maxi = 1
score_slit = 0.05

cellulates = [(5, 5), (10, 10), (20, 20)] # (cell_layers, layer_cells) # [(1, 5), (2, 5), (3, 5), (5, 5), (7, 5), (5, 10), (5, 13), (5, 15), (5, 17), (10, 10), (10, 15), (15, 15), (20, 20)]
rules = [0, 1]
aim_NT = [1, 2/5]
aim_G = [1-_ for _ in aim_NT]

#%%# Visualize Data Memo [Classification]

from ObjectiveFun import ObjectiveFunPortion # from ObjectiveFunL1L2 import ObjectiveFunPortionRule

_memo = data_paths[0]
memo = memos[_memo]
data_memo = retrieve_data_memo(memo, cellulates, rules, verbose = True)
w = dict()
tau_mini = 0
tau_maxi = 48
tau_delta = 0.25

for cellulate in cellulates:
    for rule in rules:
        rule_index = rules.index(rule)
        trajectory_set = data_memo[cellulate][rule_index]
        objective_fun = ObjectiveFunPortion(trajectory_set, species, time_mini, time_maxi, time_unit, time_delta, simulations_maxi = 10, verbose = False, show = False, rules = rules, aim_NT = aim_NT[rule_index], aim_G = aim_G[rule_index])
        classifier = objective_fun_classifier(objective_fun, tau_mini = tau_mini, tau_maxi = tau_maxi, tau_delta = tau_delta)
        x = objective_fun.tau
        y = classifier
        w.update({cellulate: y})

#%%# Visualize Data Memo [Convolution]

cellulate = cellulates[2]
clues = ['NT', 'G', 'U']
clue_labels = ['EPI', 'PRE', 'UND']
positional_information = False
verbose = False

from scipy.ndimage import convolve

degrees = (1, 2)
kernel = make_kernel(degrees, verbose)
_crux = np.ones(cellulate, int)
crux = convolve(input = _crux, weights = kernel, mode = 'constant')

neighborhood_quota = {key: None for key in clues}
hints = clues
hint_labels = clue_labels

y = w[cellulate]

for clue in clues:
    if verbose: print(f"{'~'*8} {clue_labels[clues.index(clue)]} ({clue}) {'~'*8}")
    shape = (len(hints), y[clue].shape[0], y[clue].shape[1], cellulate[0], cellulate[1])
    share = np.full(shape, np.nan)
    iota = range(y[clue].shape[0]) # simulation_index
    jota = range(y[clue].shape[1]) # tau_index
    for i in iota:
        for j in jota:
            zone = np.reshape(y[clue][i, j], cellulate)
            # print(zone)
            for hint in hints:
                # print(hint)
                hint_index = hints.index(hint)
                neighborhood = np.reshape(y[hint][i, j], cellulate)
                # print(neighborhood)
                convolution = convolve(input = neighborhood, weights = kernel, mode = 'constant') # quota
                _neighbor_count = zone*convolution
                cone = [zone == 1]
                chow = [_neighbor_count]
                neighbor_count = np.select(cone, chow, np.nan)
                neighbor_quota = neighbor_count/crux
                # print(neighbor_count)
                share[hint_index, i, j, ...] = neighbor_quota
    neighborhood_quota.update({clue: share})

# Visualize Data Memo [Probability]

hood_port_stats = ['mean', 'sand', 'cove']
hood_port_shape = (len(clues), len(hints), 1, y['U'].shape[1], len(hood_port_stats))
hood_port_alter = np.full(hood_port_shape, np.nan)

if positional_information:
    for clue in clues:
        clue_index = clues.index(clue)
        iota = range(y[clue].shape[0])
        jota = range(y[clue].shape[1])
        simulation_index = slice(min(iota), max(iota)+1) # slice(min(iota), max(iota)+1)
        tau_index = slice(min(jota), max(jota)+1) # slice(min(jota), max(jota)+1)
        alp = neighborhood_quota[clue][:, simulation_index, tau_index, ...]
        print(alp.shape)    
        for hint in hints:
            hint_index = hints.index(hint)
            print(clue, hints[hint_index])
            bet = np.nanmean(alp[hint_index], 0)
            # print(bet)
            for _ in range(bet.shape[0]):
                chi = np.nanmean(bet[_])
                dev = np.nanstd(bet[_])
                elf = dev/chi
                hood_port_alter[clue_index, hint_index, 0, _, :] = [chi, dev, elf]
                print(x[_], chi, dev, elf)
else:
    for clue in clues:
        clue_index = clues.index(clue)
        iota = range(y[clue].shape[0])
        jota = range(y[clue].shape[1])
        simulation_index = slice(min(iota), max(iota)+1) # slice(min(iota), max(iota)+1)
        tau_index = slice(min(jota), max(jota)+1) # slice(min(jota), max(jota)+1)
        alp = neighborhood_quota[clue][:, simulation_index, tau_index, ...]
        print(alp.shape)    
        for hint in hints:
            hint_index = hints.index(hint)
            print(clue, hints[hint_index])
            bet = alp[hint_index]
            # print(bet.shape)
            for _ in range(bet.shape[1]):
                chi = np.nanmean(bet[:, _, ...])
                dev = np.nanstd(bet[:, _, ...])
                elf = dev/chi
                hood_port_alter[clue_index, hint_index, 0, _, :] = [chi, dev, elf]
                print(x[_], chi, dev, elf)

#%%# Visualize Data Memo [Convolution] [Multinomial Distribution]

seed_trial = 25 # {25, 100, 400}
# cellulate = cellulates[0]
shape_trial = (data_memo[cellulate][0].shape[0], len(x), cellulate[0], cellulate[1], len(clues))
exam = np.full(shape_trial, np.nan)
_shape_trial = (data_memo[cellulate][0].shape[0], 1, cellulate[0], cellulate[1])
tempi = x.tolist()
for tempo in tempi:
    x_index = tempi.index(tempo)
    _aims = [np.mean(w[cellulate][_][:, x_index, :]) for _ in clues]
    print(f"{tempo} {'~'*8} {_aims} {'~'*8} {sum(_aims)} {'~'*8}")
    aims = np.array(_aims)
    _exam = np.random.default_rng(seed_trial).multinomial(n = 1, pvals = aims, size = _shape_trial)
    exam[:, [x_index], :, :] = _exam
trial = dict()
trial.update({'NT': exam[..., 0]})
trial.update({'G': exam[..., 1]})
trial.update({'U': exam[..., 2]})
# positional_information = False
verbose = False

from scipy.ndimage import convolve

# degrees = (0, 1, 2)
kernel = make_kernel(degrees, verbose)
_crux = np.ones(cellulate, int)
crux = convolve(input = _crux, weights = kernel, mode = 'constant')

neighborhood_quota = {key: None for key in clues}
hints = clues
hint_labels = clue_labels

y = trial

for clue in clues:
    if verbose: print(f"{'~'*8} {clue_labels[clues.index(clue)]} ({clue}) {'~'*8}")
    shape = (len(hints), y[clue].shape[0], y[clue].shape[1], cellulate[0], cellulate[1])
    share = np.full(shape, np.nan)
    iota = range(y[clue].shape[0]) # simulation_index
    jota = range(y[clue].shape[1]) # tau_index
    for i in iota:
        for j in jota:
            zone = np.reshape(y[clue][i, j], cellulate)
            # print(zone)
            for hint in hints:
                # print(hint)
                hint_index = hints.index(hint)
                neighborhood = np.reshape(y[hint][i, j], cellulate)
                # print(neighborhood)
                convolution = convolve(input = neighborhood, weights = kernel, mode = 'constant') # quota
                _neighbor_count = zone*convolution
                cone = [zone == 1]
                chow = [_neighbor_count]
                neighbor_count = np.select(cone, chow, np.nan)
                neighbor_quota = neighbor_count/crux
                # print(neighbor_count)
                share[hint_index, i, j, ...] = neighbor_quota
    neighborhood_quota.update({clue: share})

# Visualize Data Memo [Probability] [Multinomial Distribution]

# hood_port_stats = ['mean', 'sand', 'cove']
# hood_port_shape = (len(clues), len(hints), 1, y['U'].shape[1], len(hood_port_stats))
hood_port_refer = np.full(hood_port_shape, np.nan)

if positional_information:
    for clue in clues:
        clue_index = clues.index(clue)
        iota = range(y[clue].shape[0])
        jota = range(y[clue].shape[1])
        simulation_index = slice(min(iota), max(iota)+1) # slice(min(iota), max(iota)+1)
        tau_index = slice(min(jota), max(jota)+1) # slice(min(jota), max(jota)+1)
        alp = neighborhood_quota[clue][:, simulation_index, tau_index, ...]
        print(alp.shape)    
        for hint in hints:
            hint_index = hints.index(hint)
            print(clue, hints[hint_index])
            bet = np.nanmean(alp[hint_index], 0)
            # print(bet)
            for _ in range(bet.shape[0]):
                chi = np.nanmean(bet[_])
                dev = np.nanstd(bet[_])
                elf = dev/chi
                hood_port_refer[clue_index, hint_index, 0, _, :] = [chi, dev, elf]
                print(x[_], chi, dev, elf)
else:
    for clue in clues:
        clue_index = clues.index(clue)
        iota = range(y[clue].shape[0])
        jota = range(y[clue].shape[1])
        simulation_index = slice(min(iota), max(iota)+1) # slice(min(iota), max(iota)+1)
        tau_index = slice(min(jota), max(jota)+1) # slice(min(jota), max(jota)+1)
        alp = neighborhood_quota[clue][:, simulation_index, tau_index, ...]
        print(alp.shape)    
        for hint in hints:
            hint_index = hints.index(hint)
            print(clue, hints[hint_index])
            bet = alp[hint_index]
            # print(bet.shape)
            for _ in range(bet.shape[1]):
                chi = np.nanmean(bet[:, _, ...])
                dev = np.nanstd(bet[:, _, ...])
                elf = dev/chi
                hood_port_refer[clue_index, hint_index, 0, _, :] = [chi, dev, elf]
                print(x[_], chi, dev, elf)

#%%# Kernel \/ Neighborhood

def _make_kern_annotation(_kernel, degrees):
    _degrees = sorted(degrees, reverse = True)
    core = max(degrees)
    dim = 2*core+1
    kern_shape = (dim, dim)
    kern = np.full(kern_shape, np.nan)
    for degree in _degrees:
        kern[...] = np.where(_kernel[degree, ...], degree, kern)
    return kern

def make_kernel_annotation(degrees, cellulate = None, verbose = False):
    if verbose: print(f"{'<'*8} Neighborhood Composition! {'>'*8}")
    core = max(degrees)
    neighborhood_degrees = range(core+1)
    dim = 2*core+1
    kernel_shape = (dim, dim)
    kernel_temp = np.zeros(kernel_shape, int)
    _kernel_shape = (core+1, *kernel_shape)
    _kernel = np.zeros(_kernel_shape, int)
    if verbose: print(f"Kernel (Neighborhood)!\n{' '*8}Degrees {degrees}\n{' '*8}Dimension {dim}\n{' '*8}Core {core}")
    for degree in neighborhood_degrees:
        neigh_dim = 2*degree+1
        roc = core-degree
        for row in range(neigh_dim):
            mini = abs(degree-row)
            maxi = neigh_dim-1-mini
            for col in range(neigh_dim):
                if mini <= col <= maxi:
                    _kernel[degree, row+roc, col+roc] = 1
        if verbose: print(f'Degree {degree} Neighborhood!\n{_kernel[degree]}')
    for degree in neighborhood_degrees:
        if degree not in degrees:
            _degrees = range(degree+1, core+1)
            for _degree in _degrees:
                _kernel[_degree, ...] -= _kernel[degree, ...]
            _kernel[degree, ...] *= 0
    for degree in neighborhood_degrees:
        if degree in degrees:
            kernel_temp[...] += _kernel[degree, ...]
    kern = _make_kern_annotation(_kernel, degrees)
    if verbose: print(f'Neighborhood Info!\n{kern}')
    kernel = np.sign(kernel_temp).astype(int)
    if verbose: print(f"{'~'*8} Kernel (Neighborhood)! {'~'*8}\n{kernel}")
    if cellulate is None:
        kernel_annotation = None
    else:
        kernel_annotation = np.full(cellulate, np.nan)
        x = [cellulate[0]-kernel_shape[0], cellulate[0]]
        y = [cellulate[1]-kernel_shape[1], cellulate[1]]
        for i in range(x[0], x[1]):
            for j in range(y[0], y[1]):
                kernel_annotation[i, j] = kern[i-x[0], j-y[0]]
    if verbose: print(f"{'~'*8} Kernel (Annotation)! {'~'*8}\n{kernel_annotation}")
    return kernel_annotation

kernel_annotation = make_kernel_annotation(tuple(range(max(degrees)+1)), cellulate, verbose)

#%%# Create Figure!

rows = 1
cols = 4
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
axe_mat = fig.subplots(nrows = rows, ncols = cols, squeeze = False)

# Neighborhood [Time Series]

synopses = {'Mean': 1, 'Sand': 0, 'Quantiles': 0}
cocos = ['tab:blue', 'tab:orange', 'tab:olive'] # list(matplotlib.colors.TABLEAU_COLORS.keys())
cocks = ['turquoise', 'deeppink', 'forestgreen'] # list(matplotlib.colors.CSS4_COLORS.keys())
font_size_alp = 13 # {5, 7, 11, 13}
font_size_bet = 11 # {5, 7, 11, 13}
font_size_chi = 7 # {5, 7, 11, 13}
rows = 1
cols = 4
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
axe = axe_mat[0:1, 1:4]
tits = ['[B]', '[C]', '[D]'] # {['[B]', '[C]', '[D]'], ['[F]', '[G]', '[H]']}
_x = np.array([_ for _ in x if _ % 4 == 0]).astype('int')
_y = np.linspace(0, 100, 11).astype('int')
x_limes = (np.min(x), np.max(x))
y_limes = (0, 100)
_x_limes = (24, time_maxi)
# _y_limes = (-np.ceil(np.max(_z)).astype('int'), np.ceil(np.max(_z)).astype('int'))

for clue in clues:
    clue_index = clues.index(clue)
    row = clue_index // cols
    col = clue_index % cols
    _axe = axe[row, col].inset_axes([_x_limes[0]/_x_limes[1], 0.675, 1-_x_limes[0]/_x_limes[1], 0.25])
    _z = list()
    for hint in hints:
        hint_index = hints.index(hint)
        coco = cocos[hint_index]
        cock = cocks[hint_index]
        if synopses['Mean']:
            y_mean_alter = 100*hood_port_alter[clue_index, hint_index, 0, :, hood_port_stats.index('mean')]
            y_mean_refer = 100*hood_port_refer[clue_index, hint_index, 0, :, hood_port_stats.index('mean')]
            axe[row, col].plot(x, y_mean_refer, color = cock, linestyle = '--', linewidth = 2, label = hint_labels[hint_index]+' '+r'$\rho_0$', alpha = 0.75+0.125)
            axe[row, col].plot(x, y_mean_alter, color = coco, linestyle = '-', linewidth = 2, label = hint_labels[hint_index]+' '+r'$\rho_1$', alpha = 0.75+0.125)
            z = y_mean_alter-y_mean_refer
            _z.append(np.nanmax(np.abs(z[np.argwhere(x == _x_limes[0]).item():np.argwhere(x == _x_limes[1]).item()+1])))
            _axe.plot(x, z, color = coco, linestyle = '-', linewidth = 2, label = None, alpha = 0.75+0.125*0)
        if synopses['Sand']:
            y_sand_alter = 100*hood_port_alter[clue_index, hint_index, 0, :, hood_port_stats.index('sand')]
            y_sand_refer = 100*hood_port_refer[clue_index, hint_index, 0, :, hood_port_stats.index('sand')]
            axe[row, col].plot(x, y_mean_refer-y_sand_refer, color = cock, linestyle = '-.', linewidth = 1, label = None, alpha = 0.25-0.125)
            axe[row, col].plot(x, y_mean_refer+y_sand_refer, color = cock, linestyle = '-.', linewidth = 1, label = None, alpha = 0.25-0.125)
            axe[row, col].plot(x, y_mean_alter-y_sand_alter, color = coco, linestyle = ':', linewidth = 1, label = None, alpha = 0.25-0.125)
            axe[row, col].plot(x, y_mean_alter+y_sand_alter, color = coco, linestyle = ':', linewidth = 1, label = None, alpha = 0.25-0.125)
        if synopses['Quantiles']:
            pass
        axe[row, col].set_xlabel('Time [Hour]', fontsize = font_size_bet)
        if clue == 'NT': axe[row, col].set_ylabel(r'$\rho_0$     $\rho_1$', fontsize = font_size_bet)
        axe[row, col].set_xticks(ticks = _x, labels = [_ if _ % 8 == 0 else None for _ in _x], fontsize = font_size_bet, rotation = 0)
        axe[row, col].set_yticks(ticks = _y, labels = [f'{_}%' if _ % 20 == 0 else None for _ in _y] if clue == 'NT' else [None]*len(_y), fontsize = font_size_bet, rotation = 0)
        axe[row, col].set_xlim(x_limes)
        axe[row, col].set_ylim(y_limes)
        # axe[row, col].grid(alpha = 0.25, color = 'tab:gray', linestyle = '--')
        # _x_limes = (24, time_maxi)
        _y_limes = (-np.ceil(np.max(_z)).astype('int'), np.ceil(np.max(_z)).astype('int'))
        _axe.set_xticks(ticks = _x, labels = [None for _ in _x], fontsize = font_size_chi, rotation = 0)
        _y_ticks_temp = np.linspace(_y_limes[0], _y_limes[1], 2*_y_limes[1]+1).astype('int')
        _y_ticks_meet = np.linspace(_y_limes[0] if _y_limes[0] % 2 == 1 else _y_limes[0]+1, _y_limes[1] if _y_limes[1] % 2 == 1 else _y_limes[1]-1, 5)
        _y_ticks_loci = np.array([_y_ticks_temp[np.argmin(np.abs(_y_ticks_temp-_))] for _ in _y_ticks_meet if _ >= 0]).astype('int')
        _y_ticks = np.unique((-1*_y_ticks_loci, _y_ticks_loci))
        _y_labels = [f'{_}%' for _ in _y_ticks]
        _axe.set_yticks(ticks = _y_ticks, labels = _y_labels, fontsize = font_size_chi, rotation = 0)
        _axe.set_xlim(_x_limes)
        _axe.set_ylim(_y_limes)
        _axe.grid(alpha = 0.25, color = 'tab:gray', linestyle = '--')
    axe[row, col].set_title(label = clue_labels[clue_index] + ' ' + f'(Neighborhood Degree = {degrees[-1]})', fontsize = font_size_bet)
    axe[row, col].set_title(label = tits[clue_index], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
    if clue == 'U': axe[row, col].legend(fontsize = font_size_chi+3, loc = 'lower center', ncols = 3)
    _axe.set_title(label = r'$\rho_1-\rho_0$', fontsize = font_size_bet)
# plt.show()

# Neighborhood [Heat Map]

color_map = 'tab10'
tableau_cocos = list(matplotlib.colors.TABLEAU_COLORS.keys())
cocos = ['tab:blue', 'tab:orange', 'tab:olive']
# font_size_alp = 13 # {5, 7, 11, 13}
# font_size_bet = 11 # {5, 7, 11, 13}
# font_size_chi = 7 # {5, 7, 11, 13}
rows = 1
cols = 1
# fig_size = (cols*fig_size_base, rows*fig_size_base)
# fig, axe = plt.subplots(nrows = rows, ncols = cols, sharex = False, sharey = False, squeeze = False, figsize = fig_size, layout = 'constrained')
axe = axe_mat[0:1, 0:1]
tits = ['[A]'] # {['[A]'], ['[E]']}

i = 0 # simulation_index
j = int(48/time_delta) # tau_index
_heat_map_temp = {clue: w[cellulate][clue][i, j, ...].reshape(cellulate) for clue in clues}
_heat_map = {clue: tableau_cocos.index(cocos[clues.index(clue)])*_heat_map_temp[clue] for clue in clues}

for clue in clues:
    clue_index = clues.index(clue)
    heat_map_temp = _heat_map[clue]
    if clue_index == 0:
        heat_map = heat_map_temp
    else:
        heat_map += heat_map_temp

row = 0
col = 0
axe[row, col].imshow(X = heat_map, cmap = color_map, vmin = 0, vmax = 10, origin = 'upper')
ticks_major = np.arange(0, cellulate[0], 1) # cellulate[1]
ticks_minor = [_ for _ in np.arange(0, cellulate[1], 0.5) if _ % 1 == 0.5] # cellulate[0]
labels_major = [None]*len(ticks_major)
labels_minor = [None]*len(ticks_minor)
axe[row, col].set_xticks(ticks = ticks_major, labels = labels_major, minor = False)
axe[row, col].set_yticks(ticks = ticks_major, labels = labels_major, minor = False)
axe[row, col].set_xticks(ticks = ticks_minor, labels = labels_minor, minor = True)
axe[row, col].set_yticks(ticks = ticks_minor, labels = labels_minor, minor = True)
axe[row, col].spines[:].set_visible(True)
axe[row, col].spines[:].set_color('w')
axe[row, col].spines[:].set_linewidth(0.25)
axe[row, col].set_title(label = tits[0], loc = 'left', fontsize = font_size_alp, fontweight = 'bold')
axe[row, col].grid(which = 'minor', color = 'w', linewidth = 0.25, alpha = 1)
axe[row, col].set_xlabel(f'Time = {int(j*time_delta)} Hours', fontsize = font_size_bet)
# axe[row, col].set_ylabel('Cell Count', fontsize = font_size_bet)
axe[row, col].set_title(label = 'Cell Neighborhood Example', fontsize = font_size_bet)
axe[row, col].tick_params(which = 'both', bottom = False, left = False)
annotation_colors = ['k', 'w']
for i in range(0, cellulate[0]):
    for j in range(0, cellulate[1]):
        if np.isnan(kernel_annotation[i, j]):
            note = ''
        else:
            note = int(kernel_annotation[i, j])
        annotation_color = annotation_colors[int(note in degrees)]
        axe[row, col].text(x = i, y = j, s = note, ha = 'center', va = 'center', color = annotation_color, fontsize = font_size_bet, fontweight = 'book')
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig7_Alp'
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
