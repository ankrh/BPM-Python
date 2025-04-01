# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 22:05:03 2025

@author: sspelthann
"""
import utils
import numpy as np


def initialize_RI_from_function(P, hFunc, *args):
    if not callable(hFunc):
        raise TypeError("Argument 2 is the wrong type. It must be a function handle.")
    
    if hFunc.__code__.co_argcount not in [4, 5]:
        raise ValueError("The function handle must take either 4 (X, Y, n_background, nParameters) or 5 (X, Y, Z, n_background, nParameters) arguments.")
    
    nParameters = args[0] if len(args) >= 1 else []
    if not isinstance(nParameters, list):
        raise TypeError("Argument 3 is the wrong type. It must be a list.")
    
    Nz = args[1] if len(args) >= 2 else 1
    if not isinstance(Nz, int) or Nz <= 0:
        raise ValueError("Argument 4 is the wrong type. It must be a positive integer.")
    
    if Nz > 1 and hFunc.__code__.co_argcount == 4:
        raise ValueError("Nz > 1 specified, but the function handle only takes 4 arguments. It must take 5 arguments.")
    
    if hFunc.__code__.co_argcount == 4:  # 2D case
        X, Y = np.meshgrid(np.array(P.x, dtype=np.float32), np.array(P.y, dtype=np.float32), indexing='ij')
        P.n.n = np.array(hFunc(X, Y, P.n_background, nParameters), dtype=np.float32)
    else:  # 3D case
        dz_n = P.Lz / (Nz - 1) if Nz > 1 else 0
        z_n = np.linspace(0, P.Lz, Nz, dtype=np.float32)
        X, Y, Z_n = np.meshgrid(np.array(P.x, dtype=np.float32), np.array(P.y, dtype=np.float32), z_n, indexing='ij')
        n = np.array(hFunc(X, Y, Z_n, P.n_background, nParameters), dtype=np.float32)
        P.n.n = utils.trimRI(n, P.n_background)
    
    P.n.Lx = P.dx * P.n.n.shape[0]
    P.n.Ly = P.dy * P.n.n.shape[1]
    P.n.xSymmetry = P.xSymmetry
    P.n.ySymmetry = P.ySymmetry
    
    return P