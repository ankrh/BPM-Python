## Importing the necessary libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import ctypes
from ctypes import *
from scipy.io import loadmat
from numpy import ctypeslib

## Defining the complex single precision float type
class floatcomplex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), 
                ("imag", ctypes.c_float)] 

## Functions to convert between numpy arrays and ctypes arrays
def floatcomplex_numpy_to_ctypes(arr):
    flat_arr = arr.flatten()
    c_arr = (floatcomplex * flat_arr.size)()
    for i, val in enumerate(flat_arr):
        c_arr[i] = floatcomplex(val.real, val.imag)
    return c_arr

def floatcomplex_ctypes_to_numpy(ctypes_array, shape):
    np_array = np.array([(val.real + 1j * val.imag) for val in ctypes_array], dtype=np.complex64)
    return np_array.reshape(shape)  # Reshape into 2D

def float_numpy_to_ctypes(arr):
    flat_arr = arr.flatten()
    c_arr = (c_float * flat_arr.size)()
    for i, val in enumerate(flat_arr):
        c_arr[i] = c_float(val)
    return c_arr

# Load the DLL and specify the argument types
script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.abspath(os.path.join(script_dir, "../../C code/x64/Release/FDBPMpropagator.dll"))
dll = ctypes.CDLL(dll_path)
dll.entryfunc.argtypes = (POINTER(floatcomplex), # E (E1)
                          POINTER(floatcomplex), # Efinal
                          c_long, # Nx
                          c_long, # Ny
                          c_float, # dx
                          c_float, # dy
                          c_float, # dz
                          c_long, # iz_start
                          c_long, # iz_end
                          c_float, # taperPerStep
                          c_float, # twistPerStep
                          c_ubyte, # xSymmetry
                          c_ubyte, # ySymmetry
                          c_float, # d
                          c_float, # n_0
                          POINTER(floatcomplex), # n_mat (n_in)
                          POINTER(floatcomplex), # n_mat (n_out)
                          c_long, # Nx_n
                          c_long, # Ny_n
                          c_long, # Nz_n
                          c_float, # dz_n
                          c_float, # rho_e
                          c_float, # RoC
                          c_float, # bendDirection
                          c_double, # precisePower input
                          POINTER(c_double), # precisePower output
                          POINTER(c_float), # multiplier
                          floatcomplex, # ax
                          floatcomplex, # ay
                          c_ubyte # useAllCPUs
                          )  # Specify argument types
dll.entryfunc.restype = None # Specify return type

## Load an example set of inputs from a .mat file
mexparameters = loadmat('mexparameters.mat')['mexParameters'][0][0]

dx = mexparameters['dx'][0][0]
dy = mexparameters['dy'][0][0]
dz = mexparameters['dz'][0][0]
taperPerStep = mexparameters['taperPerStep'][0][0]
twistPerStep = mexparameters['twistPerStep'][0][0]
dz_n = mexparameters['dz_n'][0][0]
d = mexparameters['d'][0][0]
n_0 = mexparameters['n_0'][0][0]
ax = mexparameters['ax'][0][0]
ax = floatcomplex(ax.real, ax.imag)
ay = mexparameters['ay'][0][0]
ay = floatcomplex(ay.real, ay.imag)
useAllCPUs = mexparameters['useAllCPUs'][0][0]
RoC = mexparameters['RoC'][0][0]
rho_e = mexparameters['rho_e'][0][0]
bendDirection = mexparameters['bendDirection'][0][0]
inputPrecisePower = mexparameters['inputPrecisePower'][0][0].astype(np.float64)
xSymmetry = mexparameters['xSymmetry'][0][0]
ySymmetry = mexparameters['ySymmetry'][0][0]
iz_start = mexparameters['iz_start'][0][0]
iz_end = 100 #mexparameters['iz_end'][0][0]
multiplier = mexparameters['multiplier']
n_mat = mexparameters['n_mat']
E = loadmat('E.mat')['E']
Nx, Ny = E.shape
shape = n_mat.shape
if len(shape) == 2:
    Nx_n, Ny_n = shape
    Nz_n = 1
else:
    Nx_n, Ny_n, Nz_n = shape

## Convert the input arrays to ctypes
E_C = floatcomplex_numpy_to_ctypes(E)
Efinal_C = (floatcomplex*(Nx*Ny))()
n_mat_C = floatcomplex_numpy_to_ctypes(n_mat)
n_out_C = (floatcomplex*(Nx*Ny))()
outputPrecisePowerPtr = (c_double*1)()
multiplier_C = float_numpy_to_ctypes(multiplier)

## Call the DLL function
dll.entryfunc(E_C,
              Efinal_C,
              Nx,
              Ny,
              dx, 
              dy, 
              dz,
              iz_start, 
              iz_end,
              taperPerStep, 
              twistPerStep,
              xSymmetry, 
              ySymmetry,
              d, 
              n_0,
              n_mat_C, 
              n_out_C,
              Nx_n, 
              Ny_n, 
              Nz_n,
              dz_n, 
              rho_e, 
              RoC,
              bendDirection,
              inputPrecisePower, 
              outputPrecisePowerPtr,
              multiplier_C,
              ax,
              ay,
              useAllCPUs)

## Convert the output arrays to numpy
E_final = floatcomplex_ctypes_to_numpy(Efinal_C, (Nx, Ny))
n_out   = floatcomplex_ctypes_to_numpy(n_out_C, (Nx, Ny))

## Plot the initial and final E fields
x = dx * np.arange(-Nx/2 + 1/2, Nx/2, 1)
y = dy * np.arange(-Ny/2 + 1/2, Ny/2, 1)
X, Y = np.meshgrid(x.astype(np.float32),y.astype(np.float32), indexing='ij')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im1 = axes[0].imshow(abs(E.T)**2, extent=[x[0], x[-1], y[0], y[-1]], aspect='equal')
axes[0].set_title("Initial E field")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])
im2 = axes[1].imshow(abs(E_final.T)**2, extent=[x[0], x[-1], y[0], y[-1]], aspect='equal')
axes[1].set_title("Final E field")
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])
plt.tight_layout()
plt.show()