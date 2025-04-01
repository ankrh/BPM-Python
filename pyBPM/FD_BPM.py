# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:05:36 2025

@author: sspelthann
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def FD_BPM(E_input, dx, dz, wavelength, n, num_steps, pml_width=20):
    """
    Simulates electric field propagation using Finite Difference Beam Propagation Method (FD-BPM).
    
    Parameters:
        E_input (2D numpy array): Initial electric field distribution.
        dx (float): Spatial step in x direction.
        dz (float): Propagation step in z direction.
        wavelength (float): Wavelength of the light.
        n (2D numpy array): Refractive index distribution (real and complex).
        num_steps (int): Number of propagation steps.
        pml_width (int): Width of the PML (Perfectly Matched Layer) for boundary conditions.
    
    Returns:
        2D numpy array: Final electric field distribution after propagation.
    """
    
    # Free-space wave number
    k0 = 2 * np.pi / wavelength  
    nx, ny = E_input.shape  # Grid size
    
    # Extract real and imaginary parts of refractive index
    n_real = np.real(n)
    n_imag = np.imag(n)
    
    # Prepare the Laplacian operator (in x direction) using finite differences
    diag_main = -2.0 * np.ones(nx)
    diag_off = np.ones(nx - 1)
    laplacian_x = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csr') / dx**2

    # Initialize electric field (copy to avoid modifying input)
    E = E_input.copy()

    # Create the PML profile for boundaries (in x-direction)
    pml_profile_x = np.ones(nx)
    pml_profile_x[:pml_width] = np.exp(-np.linspace(0, 10, pml_width))  # Left boundary
    pml_profile_x[-pml_width:] = np.exp(-np.linspace(0, 10, pml_width)[::-1])  # Right boundary

    # Propagation loop for the specified number of steps (over z)
    for step in range(num_steps):
        # Compute the effective refractive index for propagation (using the complex part of n)
        n_eff = n_real[:, ny // 2] + 1j * n_imag[:, ny // 2]  # Central slice in y-direction
        
        # Calculate the potential term for the propagation matrix (taking into account complex refractive index)
        potential = k0**2 * (n_eff**2 - np.mean(n_eff**2))

        # Define the propagation operator in the x-direction (Laplacian + potential)
        operator_x = laplacian_x + diags(potential, 0, format='csr')
        
        # Identity matrix for the Crank-Nicolson method
        I = diags(np.ones(nx), 0, format='csr') 
        
        # Crank-Nicolson scheme matrices A and B
        A = I - 1j * dz / 2 * operator_x
        B = I + 1j * dz / 2 * operator_x
        
        # Solve the system using sparse linear solver
        E = spsolve(A, B @ E)
        
        # Apply PML boundary conditions (absorbing boundary at both sides)
        E[0, :] *= pml_profile_x  # Left boundary
        E[-1, :] *= pml_profile_x  # Right boundary
        
    return np.abs(E)  # Return the magnitude of the electric field after propagation

