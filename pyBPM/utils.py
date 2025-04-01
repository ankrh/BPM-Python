# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 22:08:50 2025

@author: sspelthann
"""
import numpy as np
import matplotlib.pyplot as plt

def calc_full_field(x, y, E):
    if np.sign(np.min(x)) == 0:  # ySymmetry == 2, anti-symmetric
        x = np.concatenate([-np.flip(x[1:]), x])
        E = np.concatenate([-np.flip(E[1:], axis=0), E], axis=0)
    elif np.sign(np.min(x)) == 1:  # ySymmetry == 1, symmetric
        x = np.concatenate([-np.flip(x), x])
        E = np.concatenate([np.flip(E, axis=0), E], axis=0)
    
    if np.sign(np.min(y)) == 0:  # xSymmetry == 2, anti-symmetric
        y = np.concatenate([-np.flip(y[1:]) , y])
        E = np.concatenate([-np.flip(E[:, 1:], axis=1), E], axis=1)
    elif np.sign(np.min(y)) == 1:  # xSymmetry == 1, symmetric
        y = np.concatenate([-np.flip(y), y])
        E = np.concatenate([np.flip(E, axis=1), E], axis=1)
    
    return x, y, E


def calc_full_RI(x, y, n):
    if np.sign(np.min(x)) == -1:  # ySymmetry == 0, no symmetry
        pass  # Do nothing
    elif np.sign(np.min(x)) == 0:  # ySymmetry == 2, anti-symmetric
        x = np.concatenate((-np.flip(x[1:], axis=1), x), axis=1)
        n = np.concatenate((np.flip(n[1:], axis=0), n), axis=0)
    elif np.sign(np.min(x)) == 1:  # ySymmetry == 1, symmetric
        x = np.concatenate((-np.flip(x, axis=1), x), axis=1)
        n = np.concatenate((np.flip(n, axis=0), n), axis=0)

    if np.sign(np.min(y)) == -1:  # xSymmetry == 0, no symmetry
        pass  # Do nothing
    elif np.sign(np.min(y)) == 0:  # xSymmetry == 2, anti-symmetric
        y = np.concatenate((-np.flip(y[1:], axis=1), y), axis=1)
        n = np.concatenate((np.flip(n[:, 1:], axis=1), n), axis=1)
    elif np.sign(np.min(y)) == 1:  # xSymmetry == 1, symmetric
        y = np.concatenate((-np.flip(y, axis=1), y), axis=1)
        n = np.concatenate((np.flip(n, axis=1), n), axis=1)

    return x, y, n


def find_cores(n, n_background):
    A = np.float32(n) != np.float32(n_background)

    Nx, Ny = A.shape
    B = np.zeros((Nx, Ny), dtype=int)
    n = 0
    
    for ix in range(Nx):
        on = False
        for iy in range(Ny):
            if not on and A[ix, iy]:
                n += 1
                B[ix, iy] = n
                on = True
            elif on and A[ix, iy]:
                B[ix, iy] = n
            elif on and not A[ix, iy]:
                on = False

    for iy in range(Ny):
        for ix in range(1, Nx):
            if B[ix, iy] and B[ix - 1, iy]:
                B[B == B[ix, iy]] = B[ix - 1, iy]

    _, _, ic = np.unique(B, return_inverse=True)

    coreIdxs = np.reshape(ic - 1, B.shape)
    
    return coreIdxs


def get_grid_array(Nx, dx, symmetry):
    if symmetry == 'NoSymmetry':
        x = dx * (np.arange(-Nx // 2 + 1 / 2, Nx // 2 - 1 / 2))
    elif symmetry == 'Symmetry':
        x = dx * np.arange(1 / 2, Nx - 1 / 2)
    elif symmetry == 'AntiSymmetry':
        x = dx * np.arange(0, Nx)
    return x


def get_labeled_mode_index(P, label):
    idx = next((i for i, mode in enumerate(P.modes) if mode['label'] == label), None)
    if idx is None:
        raise ValueError(f"Mode {label} not found")
    return idx


def cividisColormap():
    #TODO
    #available from matplotlib...
    pass


def GPBGYRcolormap():
    #TODO
    #not available from matplotlib or seaborn...
    pass


def inferno():
    #TODO
    #available from matplotlib...
    pass


def plotVolumetric():
    #TODO
    pass


def redrawVolumetric():
    #TODO
    pass


def setColormap():
    #TODO
    #Should be possible via matplotlib...
    pass


def test_radial_symmetry(X, Y, n, n_background, xSymmetry, ySymmetry):
    n = np.double(n) - np.double(n_background)

    if ySymmetry:
        xC = 0
    else:
        xC = np.sum(X * np.abs(n)**2) / np.sum(np.abs(n)**2)  # x-Schwerpunkt

    if xSymmetry:
        yC = 0
    else:
        yC = np.sum(Y * np.abs(n)**2) / np.sum(np.abs(n)**2)  # y-Schwerpunkt

    R = np.sqrt((X - xC)**2 + (Y - yC)**2)  # Distanzen aller Pixel vom Schwerpunkt
    sortIdxs = np.argsort(R.flatten())
    nsortedRounded = np.round(n.flatten()[sortIdxs], 5)

    monotonicity_real = np.sign(np.diff(np.real(nsortedRounded)))
    reversals_real = np.sum(np.abs(np.diff(monotonicity_real[monotonicity_real != 0])) / 2)

    monotonicity_imag = np.sign(np.diff(np.imag(nsortedRounded)))
    reversals_imag = np.sum(np.abs(np.diff(monotonicity_imag[monotonicity_imag != 0])) / 2)

    Rsorted = R.flatten()[sortIdxs]
    
    # -- PLOT FOR TESTING
    # plt.figure(201)
    # plt.plot(Rsorted, nsortedRounded)
    # plt.grid(True, which='both')
    
    # plt.figure(202)
    # plt.imshow(n.T, extent=[X[0, 0], X[-1, 0], Y[0, 0], Y[0, -1]], aspect='equal', origin='lower')
    # plt.axis('tight')
    
    # plt.figure(203)
    # plt.plot(Rsorted[:-1], np.diff(nsortedRounded))
    # plt.grid(True, which='both')
    

    radiallySymmetric = reversals_real < 5 and reversals_imag < 5  # 5 Umkehrungen sind das Maximum, das wir zulassen. Nicht-radial symmetrische Verteilungen haben viele Umkehrungen.

    return radiallySymmetric, xC, yC


def trimRI(n, n_background):
    n_background = np.float32(n_background)
    Nx, Ny = n.shape[:2]
    
    xmin = next((i for i in range(Nx) if np.any(n[i, :, :] != n_background)), None)
    xmax = next((i for i in range(Nx - 1, -1, -1) if np.any(n[i, :, :] != n_background)), None)
    ymin = next((i for i in range(Ny) if np.any(n[:, i, :] != n_background)), None)
    ymax = next((i for i in range(Ny - 1, -1, -1) if np.any(n[:, i, :] != n_background)), None)
    
    xtrim = min(xmin, Nx - xmax - 1) if xmin is not None and xmax is not None else 0
    ytrim = min(ymin, Ny - ymax - 1) if ymin is not None and ymax is not None else 0
    
    n_temp = n[xtrim:Nx - xtrim, ytrim:Ny - ytrim, ...]
    pad_shape = [(2, 2)] * len(n_temp.shape)
    pad_shape[-1] = (0, 0)  # No padding on the last dimension if 3D
    n_padded = np.pad(n_temp, pad_shape, mode='constant', constant_values=n_background)
    
    return n_padded


def update_volumetric(h_f, M):
    h_f.UserData[:-1, :-1, :-1] = M.astype(np.float32)
    h_f.UserData[-1, :, :] = h_f.UserData[-2, :, :]
    h_f.UserData[:, -1, :] = h_f.UserData[:, -2, :]
    h_f.UserData[:, :, -1] = h_f.UserData[:, :, -2]

    i = 0
    while i < len(h_f.Children):
        if hasattr(h_f.Children[i], 'Style') and h_f.Children[i].Style == 'checkbox':
            h_f.Children[i].Callback[0](h_f.Children[i], [], h_f.Children[i].Callback[1])
            break
        else:
            i += 1

