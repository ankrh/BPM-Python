# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 21:58:24 2025

@author: sspelthann
"""
import numpy as np

def initialize_E_from_function(P, hFunc, *args):
    if not callable(hFunc):
        raise TypeError("Argument 2 is of the wrong type. It must be a function handle.")
    
    if args:
        Eparameters = args[0]
        if not isinstance(Eparameters, list):
            raise TypeError("Argument 3 is wrong type. Must be a list.")
    else:
        Eparameters = []
    
    X, Y = np.meshgrid(np.array(P.x, dtype=np.float32), np.array(P.y, dtype=np.float32), indexing='ij')
    
    E = hFunc(X, Y, Eparameters)  # Call function to initialize E field
    
    power_fraction = 1 / ((1 + (P.xSymmetry != 0)) * (1 + (P.ySymmetry != 0)))  # Fraction of total power being simulated
    

    P.E.field = E / np.sqrt(np.sum(np.abs(E) ** 2) / power_fraction)
    P.E.Lx = P.Lx
    P.E.Ly = P.Ly
    P.E.xSymmetry = P.xSymmetry
    P.E.ySymmetry = P.ySymmetry
    
    return P