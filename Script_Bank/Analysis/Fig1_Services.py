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
