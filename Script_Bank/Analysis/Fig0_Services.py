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
import pickle
from Utilities import make_paras
from Utilities import simul_data_load

#%%# Retrieve Mapes

def retrieve_mapes(data_paths, acts, observers, curbs, verbose = False):
    mapes = list()
    for index in range(len(data_paths)):
        data_path = data_paths[index]
        act = acts[index]
        observe = observers[index]
        curb = curbs[index]
        path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
        tag = f'Act_{act}_Observe_{observe}_{curb}_Posterior.pkl'
        with open(path + tag, 'rb') as portfolio:
            posterior = pickle.load(portfolio)
        _mape = posterior.map().numpy()
        mape = [int(round(_mape[index], 0)) if _mape[index] > 1 else round(_mape[index], 2) for index in range(_mape.size)]
        if verbose: print(f'MAPE!\n{data_path}\n{tag}\n\t{mape}')
        mapes.append(mape)
    return mapes

#%%# Retrieve Para Set Truths

def retrieve_para_set_truths(mapes, para_key_sets, para_value_sets, para_set_modes, verbose = False):
    para_set_truths = list()
    for index in range(len(mapes)):
        mape = mapes[index]
        para_key_set = para_key_sets[index]
        para_value_set = para_value_sets[index]
        para_set_mode = para_set_modes[index]
        para_set_raw = {para_key_set[_]: (mape[_], para_value_set[_]) for _ in range(len(mape))}
        _, para_set_true = make_paras(para_set_raw, para_set_mode, verbose)
        para_set_truths.append(para_set_true)
    return para_set_truths

#%%# Retrieve Data [Cellulate /\ Memo]

def retrieve_data_cellulate(memo, cellulate, reiteration, verbose = False):
    if verbose: print(f"{' '*8}{cellulate}")
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
    tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate = (theta_set, trajectory_set)
    return data_cellulate

def retrieve_data_memo(memo, cellulates, rules, verbose = False):
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
            data_cellulate = retrieve_data_cellulate(memo, cellulate, reiteration, verbose)
            splitter = int(data_cellulate[1].shape[1]/len(rules))
            rule_split = [(rules.index(rule)*splitter, (rules.index(rule)+1)*splitter) for rule in rules]
            for rule in rules:
                rule_index = rules.index(rule)
                split = rule_split[rule_index]
                data_memo[cellulate][rule_index] = data_cellulate[1][:, split[0]:split[1]]
    return data_memo

#%%# Retrieve Data [Cellulate /\ {Auto Para Mem | Autocrine Paracrine Membrane} /\ Memo]

def retrieve_data_cellulate_auto_para_mem(memo, cellulate, auto_para_mem, reiteration, verbose = False):
    if verbose: print(f"{' '*8}{cellulate}{' '*8}{auto_para_mem}")
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
    tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate_auto_para_mem = (theta_set, trajectory_set)
    return data_cellulate_auto_para_mem

def retrieve_data_memo_auto_para_mem(memo, cellulates, auto_para_mem_set, meth = ('Auto', 'Para', 'Mem'), verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    data_memo_auto_para_mem = {(cellulate, auto_para_mem): None for cellulate in cellulates for auto_para_mem in auto_para_mem_set}
    _meth = '_'.join(map(str, meth))
    for cellulate in cellulates:
        for auto_para_mem in auto_para_mem_set:
            _auto_para_mem = '_'.join(map(str, auto_para_mem))
            reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}_{_meth}_{_auto_para_mem}'
            data_cellulate_auto_para_mem = retrieve_data_cellulate_auto_para_mem(memo, cellulate, auto_para_mem, reiteration, verbose)
            data_memo_auto_para_mem[(cellulate, auto_para_mem)] = data_cellulate_auto_para_mem[1]
    return data_memo_auto_para_mem

#%%# Retrieve Data [Sole Rule]

def retrieve_data_cellulate_sole_rule(memo, cellulate, auto_para_mem, reiteration, verbose = False):
    if verbose: print(f"{' '*8}{cellulate}{' '*8}{auto_para_mem}")
    data_path = memo['data_path']
    act = memo['act']
    observe = memo['observe']
    curb = memo['curb']
    path = os.path.dirname(os.path.realpath(__file__)) + f'/../../Data_Bank/{data_path}/Observe_{observe}/'
    tag = f'Act_{act}_Observe_{observe}_{curb}_{reiteration}'
    theta_set, trajectory_set = simul_data_load(None, path, tag, None, verbose)
    data_cellulate_auto_para_mem = (theta_set, trajectory_set)
    return data_cellulate_auto_para_mem

def retrieve_data_memo_sole_rule(memo, cellulates, auto_para_mem_set, meth = ('Auto', 'Para', 'Mem'), verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    data_memo_auto_para_mem = {(cellulate, auto_para_mem): None for cellulate in cellulates for auto_para_mem in auto_para_mem_set}
    _meth = '_'.join(map(str, meth))
    for cellulate in cellulates:
        for auto_para_mem in auto_para_mem_set:
            _auto_para_mem = '_'.join(map(str, auto_para_mem))
            reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}_{_meth}_{_auto_para_mem}'
            data_cellulate_auto_para_mem = retrieve_data_cellulate_auto_para_mem(memo, cellulate, auto_para_mem, reiteration, verbose)
            data_memo_auto_para_mem[(cellulate, auto_para_mem)] = data_cellulate_auto_para_mem[1]
    return data_memo_auto_para_mem

#%%# Retrieve Data [Initiate]

def retrieve_data_cellulate_initiate(memo, cellulate, initiate, reiteration, verbose = False):
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

def retrieve_data_memo_initiate(memo, cellulates, initiate_set, wait, verbose = False):
    if verbose: print(f"{'<'*8+' '*8}{memo['data_path']}{' '*8+'>'*8}")
    keys = list(initiate_set.keys())
    values = list(initiate_set.values())
    initiate_set_temp = list(zip(*values))
    data_memo_initiate = {(cellulate, initiate_temp): None for cellulate in cellulates for initiate_temp in initiate_set_temp}
    for cellulate in cellulates:
        for initiate_temp in initiate_set_temp:
            initiate = {keys[index]: initiate_temp[index] for index in range(len(keys))}
            _initiate = '_'.join(map(str, [f'{key}_{value}' for key, value in initiate.items()]))
            reiteration = f'Reiterate_{cellulate[0]}_{cellulate[1]}_Initiate_{_initiate}_Wait_{wait}' # {Wild, Nook, Cast}
            data_cellulate_initiate = retrieve_data_cellulate_initiate(memo, cellulate, initiate, reiteration, verbose)
            data_memo_initiate[(cellulate, initiate_temp)] = data_cellulate_initiate[1]
    return data_memo_initiate

#%%# Objective Fun Counter

def objective_fun_counter(self, **keywords):
    # Data Preprocessor! [Start]
    self.restructure_data() # Step Zero!
    tau_mini = keywords.get('tau_mini', None)
    tau_maxi = keywords.get('tau_maxi', None)
    tau_delta = keywords.get('tau_delta', None)
    self.decimate_data(tau_mini, tau_maxi, tau_delta)
    species_sieve = ['N', 'G', 'NP']
    self.sieve_data(species_sieve)
    species_comb_tram_dit = {'NT': (('N', 'NP'), np.add)} # {'GP': (('G', 0.5), np.power)}
    self.comb_tram_data(species_comb_tram_dit)
    species_sieve = ['NT', 'G']
    self.sieve_data(species_sieve)
    # Data Preprocessor! [Final]
    # Data Processor! [Start]
    threshold_NT_positive = self.positive_NT-self.cusp_NT*np.sqrt(self.positive_NT)
    threshold_NT_negative = self.negative_NT+self.cusp_NT*np.sqrt(self.negative_NT)
    threshold_G_positive = self.positive_G-self.cusp_G*np.sqrt(self.positive_G)
    threshold_G_negative = self.negative_G+self.cusp_G*np.sqrt(self.negative_G)
    NT_positive = self.data_objective[:, self.species_objective.index('NT'), :, :] > threshold_NT_positive
    NT_negative = self.data_objective[:, self.species_objective.index('NT'), :, :] < threshold_NT_negative
    G_positive = self.data_objective[:, self.species_objective.index('G'), :, :] > threshold_G_positive
    G_negative = self.data_objective[:, self.species_objective.index('G'), :, :] < threshold_G_negative
    classification = { # (NT, G) # '(+|-)(+|-)'
        '++': np.logical_and(NT_positive, G_positive),
        '+-': np.logical_and(NT_positive, G_negative),
        '-+': np.logical_and(NT_negative, G_positive),
        '--': np.logical_and(NT_negative, G_negative),
        '¿?': np.invert(np.logical_or(np.logical_and(NT_positive, G_negative), np.logical_and(NT_negative, G_positive)))
    }
    counter = dict()
    counter.update({'DP': np.count_nonzero(classification['++'], 2)})
    counter.update({'NT': np.count_nonzero(classification['+-'], 2)})
    counter.update({'G': np.count_nonzero(classification['-+'], 2)})
    counter.update({'DN': np.count_nonzero(classification['--'], 2)})
    counter.update({'U': np.count_nonzero(classification['¿?'], 2)})
    return counter

#%%# Bootstrap Trajectory Set

def bootstrap_trajectory_set(trajectory_set, cellulate_duo, species, time_mini, time_maxi, time_delta, seed = None, verbose = False):
    cellulate_root = cellulate_duo[0]
    cellulate_leaf = cellulate_duo[1]
    simulations_root = trajectory_set.shape[0]
    simulations_leaf = trajectory_set.shape[0]
    size_root = cellulate_root[0]*cellulate_root[1]
    size_leaf = cellulate_leaf[0]*cellulate_leaf[1]
    shape_root = (simulations_root, len(species), int((time_maxi+1)/time_delta), size_root)
    root = trajectory_set.reshape(shape_root)
    ratio = int(size_root/size_leaf)
    scope = int(simulations_root/(ratio))
    collect = np.random.default_rng(seed).choice(a = simulations_root, size = scope, replace = False)
    _leaf = root[collect, ...]
    shape_leaf = (simulations_leaf, len(species), int((time_maxi+1)/time_delta), size_leaf)
    leaf = np.zeros(shape_leaf)
    jota = [(_*size_leaf, (_+1)*size_leaf) for _ in range(ratio)]
    if verbose: print(f'Jota! {ratio}\n\t[{jota[0]}, ..., {jota[-1]}]')
    iota = [(_*ratio, (_+1)*ratio) for _ in range(scope)]
    if verbose: print(f'Iota! {scope}\n\t[{iota[0]}, ..., {iota[-1]}]')
    for i in range(len(iota)):
        _bootstrap = _leaf[[i], ...]
        for j in range(len(jota)):
            bootstrap_temp = _bootstrap[..., jota[j][0]:jota[j][1]]
            if j == 0:
                bootstrap = bootstrap_temp
            else:
                bootstrap = np.concatenate((bootstrap, bootstrap_temp))
        leaf[iota[i][0]:iota[i][1], ...] = bootstrap
    trajectory_set_bootstrap = leaf.reshape((simulations_leaf, -1))
    return trajectory_set_bootstrap

#%%# Make Tau Tally

def make_tau_tally(cell_count, cell_tally, tau, correct = None, verbose = False):
    tau_where = np.argmax(a = cell_count >= cell_tally, axis = 1) # tau_where = np.argmin(a = cell_count < cell_tally, axis = 1)
    tau_tally_collect = np.unique(np.nonzero(a = cell_count >= cell_tally)[0])
    if verbose: print(f'Collect! {tau_tally_collect.size}')
    tau_tally_correct = np.setdiff1d(np.arange(cell_count.shape[0]), tau_tally_collect)
    _tau_tally = tau[tau_where]
    if correct is not None and tau_tally_correct.size > 0:
        if verbose: print(f'Correct! {tau_tally_correct.size} ~ {correct}')
        tau_tally = _tau_tally
        tau_tally[tau_tally_correct] = correct
    else:
        tau_tally = _tau_tally[tau_tally_collect]
    return tau_tally

#%%# Assistance

def razor(numb):
    if numb == 0 or numb == 1:
        shave = numb
    elif numb > 1:
        shave = np.ceil(numb)
    elif 0 < numb < 1:
        log = np.log10(numb)
        floor = np.floor(log)
        ceil = np.ceil(numb*np.power(10, -floor))
        shave = np.round(ceil*np.power(10, floor), int(abs(floor)))
    else: # numb < 0
        raise RuntimeError('Oops! Negative numbers are not acceptable!')
    return shave

def diver(numb, slit = 0.05):
    log = np.log10(numb)
    ceil = np.ceil(log)
    divo = slit*np.power(10, ceil)
    return divo

def mull(numb, divo = 1):
    mess = "The variable 'divo' must be greater than '0'!"
    check = divo > 0
    assert check, mess
    if numb == 0 or numb == 1:
        mule = numb
    elif numb > 1:
        _mule = np.ceil(numb)
        remainder = int(np.invert(np.isclose(_mule % divo, 0)))
        mule = divo * ((_mule // divo) + remainder)
    elif 0 < numb < 1:
        _log = np.log10(divo)
        _floor = np.floor(_log)
        log = np.log10(numb)
        floor = np.floor(log)+_floor
        ceil = np.ceil(numb*np.power(10, -floor))
        _mule = np.round(ceil*np.power(10, floor), int(abs(floor)))
        remainder = int(np.invert(np.isclose(_mule % divo, 0)))
        mule = np.round(divo * ((_mule // divo) + remainder), int(abs(floor)))
    else: # numb < 0
        raise RuntimeError('Oops! Negative numbers are not acceptable!')
    return mule

def stepper(numb, slit = 0.05):
    _log = np.log10(slit)
    _floor = np.floor(_log)-1
    log = np.log10(numb)
    floor = np.floor(log)+_floor
    _step = np.ceil(numb/np.power(10, floor))
    step = np.round(_step*slit*np.power(10, floor), int(abs(floor)))
    return step

#%%# Objective Fun Classifier

def objective_fun_classifier(self, **keywords):
    # Data Preprocessor! [Start]
    self.restructure_data() # Step Zero!
    tau_mini = keywords.get('tau_mini', None)
    tau_maxi = keywords.get('tau_maxi', None)
    tau_delta = keywords.get('tau_delta', None)
    self.decimate_data(tau_mini, tau_maxi, tau_delta)
    species_sieve = ['N', 'G', 'NP']
    self.sieve_data(species_sieve)
    species_comb_tram_dit = {'NT': (('N', 'NP'), np.add)} # {'GP': (('G', 0.5), np.power)}
    self.comb_tram_data(species_comb_tram_dit)
    species_sieve = ['NT', 'G']
    self.sieve_data(species_sieve)
    # Data Preprocessor! [Final]
    # Data Processor! [Start]
    threshold_NT_positive = self.positive_NT-self.cusp_NT*np.sqrt(self.positive_NT)
    threshold_NT_negative = self.negative_NT+self.cusp_NT*np.sqrt(self.negative_NT)
    threshold_G_positive = self.positive_G-self.cusp_G*np.sqrt(self.positive_G)
    threshold_G_negative = self.negative_G+self.cusp_G*np.sqrt(self.negative_G)
    NT_positive = self.data_objective[:, self.species_objective.index('NT'), :, :] > threshold_NT_positive
    NT_negative = self.data_objective[:, self.species_objective.index('NT'), :, :] < threshold_NT_negative
    G_positive = self.data_objective[:, self.species_objective.index('G'), :, :] > threshold_G_positive
    G_negative = self.data_objective[:, self.species_objective.index('G'), :, :] < threshold_G_negative
    classifier = { # (NT, G) # '(+|-)(+|-)'
        'DP': np.logical_and(NT_positive, G_positive).astype(int),
        'NT': np.logical_and(NT_positive, G_negative).astype(int),
        'G': np.logical_and(NT_negative, G_positive).astype(int),
        'DN': np.logical_and(NT_negative, G_negative).astype(int),
        'U': np.invert(np.logical_or(np.logical_and(NT_positive, G_negative), np.logical_and(NT_negative, G_positive))).astype(int)
    }
    return classifier

#%%# Kernel \/ Neighborhood

def _make_kern(_kernel, degrees):
    _degrees = sorted(degrees, reverse = True)
    core = max(degrees)
    dim = 2*core+1
    kern_shape = (dim, dim)
    kern = np.full(kern_shape, np.nan)
    for degree in _degrees:
        kern[...] = np.where(_kernel[degree, ...], degree, kern)
    return kern

def make_kernel(degrees, verbose = False):
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
    kern = _make_kern(_kernel, degrees)
    if verbose: print(f'Neighborhood Info!\n{kern}')
    kernel = np.sign(kernel_temp).astype(int)
    if verbose: print(f"{'~'*8} Kernel (Neighborhood)! {'~'*8}\n{kernel}")
    return kernel

#%%# Objective Fun Cellular Stats

def objective_fun_cellular_stats(self, **keywords):
    cellular_stats = dict()
    # Data Preprocessor! [Start]
    self.restructure_data() # Step Zero!
    tau_mini = keywords.get('tau_mini', None)
    tau_maxi = keywords.get('tau_maxi', None)
    tau_delta = keywords.get('tau_delta', None)
    self.decimate_data(tau_mini, tau_maxi, tau_delta)
    species_comb_tram_dit = {'FCM': (('FC', 'FM'), np.add), 'FDM': (('FD', 2), np.multiply)}
    self.comb_tram_data(species_comb_tram_dit)
    species_comb_tram_dit = {'NT': (('N', 'NP'), np.add), 'FT': (('FCM', 'FDM'), np.add)}
    self.comb_tram_data(species_comb_tram_dit)
    species_sieve = ['NT', 'G', 'FT'] # species_selection
    self.sieve_data(species_sieve)
    # Data Preprocessor! [Final]
    # Data Processor! [Start]
    spas = species_sieve
    for spa in spas:
        cellular_stats.update({spa: self.data_objective[:, self.species_objective.index(spa), :, :]})
    # Data Processor! [Final]
    return cellular_stats
