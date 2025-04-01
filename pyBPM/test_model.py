# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:45:06 2025

@author: sspelthann
"""

from model import BPMModel

# -- Ensure BPMModel initializes without errors
model = BPMModel() 
#check



# -- Test if computed properties return correct values
model = BPMModel(Lx_main=30e-6, Ly_main=30e-6, Nx_main=1500, Ny_main=1500, Lz=1e-3, dz_target=5e-7)

print("dx:", model.dx())  # Should be Lx_main / Nx_main
print("dy:", model.dy())  # Should be Ly_main / Ny_main
print("dz:", model.dz())  # Should be Lz / Nz
print("Nx:", model.Nx())  # Should consider padfactor
print("Ny:", model.Ny())  # Should consider padfactor
print("Nz:", model.Nz())  # Should be at least `updates`
#check


# -- Ensure the spatial grid arrays are computed correctly
x_grid = model.x()
y_grid = model.y()

print("x array shape:", x_grid.shape)
print("y array shape:", y_grid.shape)
print("x min, max:", x_grid[0], x_grid[-1])
print("y min, max:", y_grid[0], y_grid[-1])
#check



model.finalizeVideo()  # Should print "Closing video handle (placeholder function)"
model.FD_BPM()         # Should not raise an error (yet to be implemented)
model.FFT_BPM()        # Should not raise an error
#check
