# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 21:55:42 2025

@author: sspelthann
"""
def get_labeled_mode_index(P, label):
    labels = [mode.label for mode in P.modes]
    
    try:
        return labels.index(label)
    except ValueError:
        raise ValueError(f"Mode {label} not found")