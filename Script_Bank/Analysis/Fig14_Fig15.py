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
from copy import deepcopy # import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
fig_resolution = 500 # {250, 375, 500, 750, 1000}
plt.rcParams['figure.dpi'] = fig_resolution
plt.rcParams['savefig.dpi'] = fig_resolution
fig_size_base = 10 # {2.5, 3.75, 5, 7.5, 10}

#%%# Retrieve Mapes

from Utilities import posterior_appraisal_selection

def retrieve_posteriors_mapes(data_paths, acts, observers, curbs, verbose = False):
    if verbose:
        tip = 's' if len(data_paths) > 1 else ''
        print(f"Info!\n\tWe will retrieve {len(data_paths)} {'posterior' + tip} and {len(data_paths)} {'mape' + tip}!")
    posteriors = list()
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
        if verbose: print(f'Posterior MAPE!\n{data_path}\n{_tag}\n\t{mape}')
        posteriors.append(posterior)
        mapes.append(mape)
        ret = (posteriors, mapes)
    return ret

#%%# Data Preparation

from Fig0_Services import retrieve_para_set_truths

data_paths = ['Shallow_Grid_1_N_Link', 'Shallow_Grid_1_Rule_Kern'] # data_path
nooks = [0, 0] # nook
acts = [7, 3] # act
observers = [1, 1] # observe
curbs = ['Mid', 'Mid'] # curb
posteriors, mapes = retrieve_posteriors_mapes(data_paths, acts, observers, curbs, verbose = True) # mape
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

#%%# Posterior Synthesizer

from sbi import analysis

def synthesizer_post(posterior = None, observation = None, posterior_sample_shape = None, parameter_set_true = None, mape_calculate = False, fig_size = (5, 5), verbose = False, **keywords):
    if posterior is None:
        mess = 'Oops! Please, if we wish to synthesize some simulator samples, then we need a posterior distribution!'
        raise RuntimeError(mess)
    check = isinstance(posterior.map(), torch.Tensor)
    mess = "Please, we must provide a valid 'posterior' object!\First, we must execute 'InferenceProd.synthesizer'!"
    assert check, mess
    paras = list(parameter_set_true.keys())
    para_span = torch.tensor([0, 1])
    if observation is None:
        observation = posterior.default_x
    posterior_samples = keywords.get('posterior_samples', None)
    if posterior_samples is None:
        posterior_samples = posterior.sample(sample_shape = posterior_sample_shape, x = observation)
    card = len(paras)
    theta_median = torch.quantile(posterior_samples, 0.5, 0)
    if mape_calculate:
        posterior.set_default_x(observation)
        posterior.map(num_init_samples = posterior_sample_shape[0])
    theta_mape = posterior.map()
    theta = {'median': theta_median, 'mape': theta_mape}
    minis = torch.floor(torch.min(posterior_samples, 0).values)
    maxis = torch.ceil(torch.max(posterior_samples, 0).values)
    check = torch.tensor([para_span[0] <= minis[index] and para_span[1] >= maxis[index] for index in range(posterior_samples.shape[1])])
    para_key_caste = keywords.get('para_key_caste', ...)
    para_key_elect = keywords.get('para_key_elect', None)
    para_key_label = keywords.get('para_key_label', paras) # para_key_label = [f"{int(np.round(theta['mape'].tolist()[index], 0)) if theta['mape'][index] >= 1 else np.round(theta['mape'].tolist()[index], 2)}\n{_para_key_label[index]}" for index in range(posterior_samples.shape[1])]
    if verbose:
        print(f'Caste\n\t{para_key_caste}\nElect\n\t{para_key_elect}\nLabel\n\t{[para_key_label_set[para_key_index] for para_key_index in para_key_caste]}')
    if torch.all(check):
        limes = [para_span.tolist()]*card
    else:
        para_value_set = keywords.get('para_value_set', None)
        if para_value_set is None:
            limes = [[minis[index], maxis[index]] for index in range(posterior_samples.shape[1])]
        else:
            limes = [[para_value_set[index][0], para_value_set[index][1]] for index in range(posterior_samples.shape[1])]
    spots = [[lime[0], lime[1], (lime[1]+lime[0])/2] for lime in limes]
    fig = keywords.get('fig', None)
    axes = keywords.get('axes', None)
    mark_size = keywords.get('mark_size', 7)
    points = [list(theta['mape'])]
    points_colors = ['magenta', 'blue'][0:len(points)]
    chart = analysis.pairplot(samples = posterior_samples, points = points, limits = limes, subset = para_key_caste, upper = 'hist', diag = 'hist', figsize = fig_size, labels = para_key_label, ticks = spots, points_colors = points_colors, fig = fig, axes = axes, points_offdiag = {'markersize': mark_size})
    fig_coda = keywords.get('fig_coda', None)
    axes_coda = keywords.get('axes_coda', None)
    chart_coda = analysis.conditional_pairplot(density = posterior, condition = theta['mape'], limits = limes, points = points, subset = para_key_caste, resolution = 250, figsize = fig_size, labels = para_key_label, ticks = spots, points_colors = points_colors, fig = fig_coda, axes = axes_coda, points_offdiag = {'markersize': mark_size})
    if verbose:
        print(posterior)
        print(f"MAPE\n\t{theta['mape'][para_key_caste]}")
        print(f'\n\t{para_key_elect}')
    ret = (chart, chart_coda)
    return ret

def cure_axe_mat(axe_mat = None, para_key_caste = None, para_value_set = None, mape = None, **keywords):
    cure = keywords.get('cure', set())
    axe_mat_rows, axe_mat_cols = axe_mat.shape
    _ave = [(para_value[1]+para_value[0])/2 for para_value in para_value_set]
    ave = [int(np.round(para_value, 0)) if para_value > 1 else np.round(para_value, 2) for para_value in _ave]
    if 'tick' in cure:
        font_size_tick = keywords.get('font_size_tick', 13)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_row != axe_mat_col:
                    continue
                else:
                    axe_mat_index = axe_mat_row # axe_mat_index = axe_mat_col
                    axe_mat_caste = para_key_caste[axe_mat_index]
                    spots = [para_value_set[axe_mat_caste][0], ave[axe_mat_caste], para_value_set[axe_mat_caste][1]]
                    axe_mat[axe_mat_row, axe_mat_col].set_xticks(ticks = spots, labels = spots, fontsize = font_size_tick)
    if 'label' in cure:
        font_size_label = keywords.get('font_size_label', 17)
        for axe_mat_row in range(axe_mat_rows):
            for axe_mat_col in range(axe_mat_cols):
                if axe_mat_col > axe_mat_row:
                    axe_mat_index = axe_mat_col # axe_mat_index = axe_mat_row
                    axe_mat_caste = para_key_caste[axe_mat_index]
                    x_label = mape[axe_mat_caste]
                    y_label = mape[para_key_caste[axe_mat_row]]
                    axe_mat[axe_mat_row, axe_mat_col].set_xlabel(xlabel = x_label, fontsize = font_size_label)
                    axe_mat[axe_mat_row, axe_mat_col].set_ylabel(ylabel = y_label, fontsize = font_size_label)
                else:
                    continue
    return axe_mat

#%%# Synthetic Creationism

lot = 0 # {0, 1}

posterior = posteriors[lot]
para_key_set = para_key_sets[lot]
para_value_set = para_value_sets[lot]
para_set_mode = para_set_modes[lot]
para_set_true = para_set_truths[lot]

para_key_label_sets = [ # para_key_label_set
    [r'$\mathit{Nanog}$_NANOG', r'$\mathit{Gata6}$_GATA6', r'$\mathit{Fgf4}$_NANOG', r'$\mathit{Gata6}$_A-ERK', r'$\mathit{Gata6}$_NANOG', r'$\mathit{Nanog}$_GATA6', r'$\mathit{Fgf4}$_GATA6', r'$\mathit{Nanog}$_A-ERK',
    'Mean Initial\nmRNA Count', 'Mean Initial\nPROTEIN Count',
    r'$\tau_{\mathrm{d},\mathrm{FGF4}}$', r'$\tau_{\mathrm{d},\mathrm{M‐FGFR‐FGF4}}$',
    r'$\tau_{\mathrm{sig}}$', r'$\tau_{\mathrm{exchange}}$',
    r'$\tau_{\mathrm{pho},\mathrm{ERK}}$', r'$\tau_{\mathrm{doh},\mathrm{ERK}}$',
    r'$\tau_{\mathrm{pho},\mathrm{NANOG}}$', r'$\tau_{\mathrm{doh},\mathrm{NANOG}}$',
    r'$\chi_{\mathrm{auto}}$'],
    [r'$\mathit{Nanog}$_NANOG', r'$\mathit{Gata6}$_GATA6', r'$\mathit{Gata6}$_NANOG', r'$\mathit{Nanog}$_GATA6']
]

para_key_elect_sets = [ # para_key_elect_set
    [
        ['N_N', 'G_G', 'G_N', 'N_G'],
        ['FC_N', 'G_EA', 'FC_G', 'N_EA'],
        ['tau_C', 'tau_M', 'chi_auto'],
        ['tau_ef_EA', 'tau_eb_EA', 'tau_pf_NP', 'tau_pb_NP'],
        ['MRNA', 'PRO', 'td_FC', 'td_FM']
    ],
    [
        ['N_N', 'G_G', 'G_N', 'N_G']
    ]
]

para_key_caste_sets = list() # para_key_caste_set
for para_key_set in para_key_sets:
    para_key_set_index = para_key_sets.index(para_key_set)
    para_key_elect_set = para_key_elect_sets[para_key_set_index]
    para_key_caste_set = list()
    for para_key_elect in para_key_elect_set:
        para_key_caste = [para_key_set.index(para_key) for para_key in para_key_elect]
        para_key_caste_set.append(para_key_caste)
    para_key_caste_sets.append(para_key_caste_set)

para_key_label_set = para_key_label_sets[lot]
para_key_elect_set = para_key_elect_sets[lot]
para_key_caste_set = para_key_caste_sets[lot]

#%%# Synthetic Plot! (Only Lot = 0!)

if lot != 0:
    mess = "Oops! This script section is only compatible with the data set 0!"
    raise RuntimeError(mess)

rows = 2
cols = 5
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig_size_panel = (fig_size_base, fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
fig_mat = fig.subfigures(nrows = rows, ncols = cols, squeeze = False)
verbose = True

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 31 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
mark_size_alp = 13 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}

font_size_tick = 15
font_size_label = 19
plt.rcParams['xtick.labelsize'] = font_size_tick
plt.rcParams['axes.labelsize'] = font_size_label
plt.rcParams['axes.labelweight'] = 'normal'

tits_alp = ['[A]', '[B]', '[C]', '[D]', '[E]']
tits_bet = ['[F]', '[G]', '[H]', '[I]', '[J]']

mape = [int(np.round(numb, 0)) if numb > 1 else np.round(numb, 2) for numb in posterior.map().tolist()]
cure = {'tick', 'label'}

for para_key_elect_index in range(len(para_key_elect_set)):
    print(f"{'~'*2*8}Elect Index {para_key_elect_index}{'~'*2*8}")
    para_key_caste = para_key_caste_set[para_key_elect_index]
    para_key_elect = para_key_elect_set[para_key_elect_index]
    para_key_label = para_key_label_set
    row_plan = para_key_elect_index // cols
    col_plan = para_key_elect_index % cols
    fig_vet = fig_mat[row_plan, col_plan]
    axe_mat = fig_vet.subplots(nrows = len(para_key_caste), ncols = len(para_key_caste), squeeze = False, subplot_kw = {'box_aspect': 1})
    fig_vet_coda = fig_mat[row_plan+1, col_plan]
    axe_mat_coda = fig_vet_coda.subplots(nrows = len(para_key_caste), ncols = len(para_key_caste), squeeze = False, subplot_kw = {'box_aspect': 1})
    print(f'Plan\n\t({row_plan}, {col_plan}) ~ ({row_plan+1}, {col_plan})')
    chart, chart_coda = synthesizer_post(posterior = posterior, observation = None, posterior_sample_shape = tuple([250000]), parameter_set_true = para_set_true, mape_calculate = False, fig_size = fig_size_panel, verbose = verbose, para_value_set = para_value_set, para_key_caste = para_key_caste, para_key_elect = para_key_elect, para_key_label = para_key_label, fig = fig_vet, axes = axe_mat, fig_coda = fig_vet_coda, axes_coda = axe_mat_coda, mark_size = mark_size_alp)
    tit_alp = tits_alp[para_key_elect_index]
    tit_bet = tits_bet[para_key_elect_index]
    fig_vet.suptitle(t = tit_alp, x = 0.025, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_chi, fontweight = 'bold')
    fig_vet_coda.suptitle(t = tit_bet, x = 0.025, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_chi, fontweight = 'bold')
    axe_mat = cure_axe_mat(axe_mat, para_key_caste, para_value_set, mape, font_size_tick = font_size_tick, font_size_label = font_size_tick, cure = cure)
    axe_mat_coda = cure_axe_mat(axe_mat_coda, para_key_caste, para_value_set, mape, font_size_tick = font_size_tick, font_size_label = font_size_tick, cure = cure)
plt.show()

#%%# Synthetic Plot! (Only Lot = 1!)

if lot != 1:
    mess = "Oops! This script section is only compatible with the data set 1!"
    raise RuntimeError(mess)

rows = 1
cols = 2
fig_size = (cols*fig_size_base, rows*fig_size_base)
fig_size_panel = (fig_size_base, fig_size_base)
fig = plt.figure(figsize = fig_size, layout = "constrained")
fig_mat = fig.subfigures(nrows = rows, ncols = cols, squeeze = False)
verbose = True

font_size_alp = 23 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_bet = 17 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
font_size_chi = 31 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}
mark_size_alp = 13 # {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53}

font_size_tick = 15
font_size_label = 19
plt.rcParams['xtick.labelsize'] = font_size_tick
plt.rcParams['axes.labelsize'] = font_size_label
plt.rcParams['axes.labelweight'] = 'normal'

tits_alp = ['[A]']
tits_bet = ['[B]']

mape = [int(np.round(numb, 0)) if numb > 1 else np.round(numb, 2) for numb in posterior.map().tolist()]
cure = {'tick', 'label'}

for para_key_elect_index in range(len(para_key_elect_set)):
    print(f"{'~'*2*8}Elect Index {para_key_elect_index}{'~'*2*8}")
    para_key_caste = para_key_caste_set[para_key_elect_index]
    para_key_elect = para_key_elect_set[para_key_elect_index]
    para_key_label = para_key_label_set
    row_plan = para_key_elect_index // cols
    col_plan = para_key_elect_index % cols
    fig_vet = fig_mat[row_plan, col_plan]
    axe_mat = fig_vet.subplots(nrows = len(para_key_caste), ncols = len(para_key_caste), squeeze = False, subplot_kw = {'box_aspect': 1})
    fig_vet_coda = fig_mat[row_plan, col_plan+1]
    axe_mat_coda = fig_vet_coda.subplots(nrows = len(para_key_caste), ncols = len(para_key_caste), squeeze = False, subplot_kw = {'box_aspect': 1})
    print(f'Plan\n\t({row_plan}, {col_plan}) ~ ({row_plan+1}, {col_plan})')
    chart, chart_coda = synthesizer_post(posterior = posterior, observation = None, posterior_sample_shape = tuple([250000]), parameter_set_true = para_set_true, mape_calculate = False, fig_size = fig_size_panel, verbose = verbose, para_value_set = para_value_set, para_key_caste = para_key_caste, para_key_elect = para_key_elect, para_key_label = para_key_label, fig = fig_vet, axes = axe_mat, fig_coda = fig_vet_coda, axes_coda = axe_mat_coda, mark_size = mark_size_alp)
    tit_alp = tits_alp[para_key_elect_index]
    tit_bet = tits_bet[para_key_elect_index]
    fig_vet.suptitle(t = tit_alp, x = 0.025, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_chi, fontweight = 'bold')
    fig_vet_coda.suptitle(t = tit_bet, x = 0.025, y = 1.025, ha = 'center', va = 'center', fontsize = font_size_chi, fontweight = 'bold')
    axe_mat = cure_axe_mat(axe_mat, para_key_caste, para_value_set, mape, font_size_tick = font_size_tick, font_size_label = font_size_tick, cure = cure)
    axe_mat_coda = cure_axe_mat(axe_mat_coda, para_key_caste, para_value_set, mape, font_size_tick = font_size_tick, font_size_label = font_size_tick, cure = cure)
plt.show()

#%%# Save Fig!

fig_path = os.path.dirname(os.path.realpath(__file__))
fig_nick = 'Fig14_Fig15'
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
