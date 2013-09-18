# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:27:40 2013

@author: hok1
"""

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
