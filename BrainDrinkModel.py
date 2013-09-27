# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:27:40 2013

@author: hok1
"""

import numpy as np

states = ('Energetic', 'Awake', 'Normal', 'Tired', 'Sleepy')
drinks = ('Coffee', 'Tea', 'Coke', 'Decaf', 'Juice', 'Beer', 'Wine')

transition_probabilities = {
    'Energetic' : {'Energetic': 0.1, 'Awake': 0.6, 'Normal': 0.2, 'Tired': 0.08, 'Sleepy': 0.02},
    'Awake': {'Energetic': 0.05, 'Awake': 0.3, 'Normal': 0.5, 'Tired': 0.1, 'Sleepy': 0.05},
    'Normal': {'Energetic': 0.05, 'Awake': 0.05, 'Normal': 0.4, 'Tired': 0.4, 'Sleepy': 0.1},
    'Tired': {'Energetic': 0.01, 'Awake': 0.04, 'Normal': 0.1, 'Tired': 0.45, 'Sleepy': 0.4},
    'Sleepy': {'Energetic': 0.35, 'Awake': 0.05, 'Normal': 0.05, 'Tired': 0.15, 'Sleepy': 0.4}
}

emission_probabilities = {
    'Energetic' : {'Coffee': 0.1, 'Tea': 0.1, 'Coke': 0.3, 'Decaf': 0.05, 'Juice': 0.3, 'Beer': 0.1, 'Wine': 0.05},
    'Awake': {'Coffee': 0.1, 'Tea': 0.1, 'Coke': 0.4, 'Decaf': 0.05, 'Juice': 0.2, 'Beer': 0.05, 'Wine': 0.1},
    'Normal': {'Coffee': 0.15, 'Tea': 0.15, 'Coke': 0.15, 'Decaf': 0.1, 'Juice': 0.2, 'Beer': 0.1, 'Wine': 0.15},
    'Tired': {'Coffee': 0.4, 'Tea': 0.3, 'Coke': 0.1, 'Decaf': 0.04, 'Juice': 0.01, 'Beer': 0.1, 'Wine': 0.05},
    'Sleepy': {'Coffee': 0.6, 'Tea': 0.2, 'Coke': 0.025, 'Decaf': 0.0, 'Juice': 0.025, 'Beer': 0.125, 'Wine': 0.025}
}

initial_probabilities = {'Energetic':0.3, 'Awake':0.2, 'Normal':0.2, 'Tired':0.15, 'Sleepy':0.15}

def give_state(dict_probs, rndnum):
    cum_prob = 0.0
    for state in dict_probs.keys():
        cum_prob += dict_probs[state]
        if rndnum < cum_prob:
            return state
    return dict_probs.keys()[-1]

def simulate_state_sequence(num_steps):
    rndnums = np.random.uniform(size=num_steps)
    state_seq = []
    given_state = give_state(initial_probabilities, rndnums[0])
    state_seq.append(given_state)
    for rndnum in rndnums[1:]:
        given_state = give_state(transition_probabilities[given_state], rndnum)
        state_seq.append(given_state)    
    return state_seq
    
def simulate_observed_sequence(state_seq):
    rndnums = np.random.uniform(size=len(state_seq))
    return [give_state(emission_probabilities[state], rndnum) 
                       for state, rndnum in zip(state_seq, rndnums)]
    
def prob_observed_sequence_forwardcache(observed_seq):
    matrix = []
    alphas = {}
    for state in states:
        alphas[state] = initial_probabilities[state]*emission_probabilities[state][observed_seq[0]]
    matrix.append(alphas)
    for t in range(1, len(observed_seq)):
        alphas = {}
        for state in states:
            alphas[state] = sum([matrix[t-1][previous_state]*transition_probabilities[previous_state][state]*emission_probabilities[state][observed_seq[t]] for previous_state in matrix[t-1]])
        matrix.append(alphas)
    return sum(matrix[-1].values())
    
def prob_observed_sequence_backwardcache(observed_seq):
    matrix = []
    betas = {}
    for state in states:
        betas[state] = 1.0
    matrix.append(betas)
    for t in range(len(observed_seq)-1, -1, -1):
        betas = {}
        for state in states:
            betas[state] = sum([matrix[0][subsequent_state]*transition_probabilities[state][subsequent_state]*emission_probabilities[state][observed_seq[t]] for subsequent_state in matrix[0]])
        matrix = [betas] + matrix
    prob = 0.0
    for state in states:
        prob += matrix[0][state]*initial_probabilities[state]
    return prob
    
def most_probably_state_viterbi(observed_seq):
    deltamatrix = []
    deltas = {}
    psimatrix = []
    psis = {}
    for state in states:
        deltas[state] = initial_probabilities[state]*emission_probabilities[state][observed_seq[0]]
        psis[state] = ''
    deltamatrix.append(deltas)
    psimatrix.append(psis)
    for t in range(1, len(observed_seq)):
        deltas = {}
        psis = {}
        for state in states:
            tuples = [(previous_state, deltamatrix[t-1][previous_state]*transition_probabilities[previous_state][state]*emission_probabilities[state][observed_seq[t]]) for previous_state in deltamatrix[t-1]]
            psis[state], deltas[state] = max(tuples, key=lambda item: item[1])
        deltamatrix.append(deltas)
        psimatrix.append(psis)
    most_probable_state_seq = [max(deltamatrix[-1].items(), key=lambda item: item[1])[0]]
    for t in range(len(observed_seq)-1, 0, -1):
        most_probable_state_seq = [psimatrix[t][most_probable_state_seq[0]]] + most_probable_state_seq
    return most_probable_state_seq
    
if __name__ == '__main__':
    state_seq = simulate_state_sequence(30)
    observed_seq = simulate_observed_sequence(state_seq)
    print state_seq
    print observed_seq
    print prob_observed_sequence_forwardcache(observed_seq)
    print prob_observed_sequence_backwardcache(observed_seq)
    print most_probably_state_viterbi(observed_seq)
