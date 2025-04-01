# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:37:23 2025

@author: sspel
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class BPMModel:
    # Visualization parameters
    name: str = field(default_factory=lambda: f"BPM-Python model {np.datetime64('now')}")
    figTitle: str = ""
    figNum: int = 1
    updates: int = 50
    plotEmax: float = 0.0
    plotZoom: float = 1.0
    storeE3D: bool = False
    saveVideo: bool = False
    intensityColormap: str = 'GPBGYR'
    phaseColormap: str = 'HSV'
    nColormap: str = 'Parula'
    n_colorlimits: tuple = (0, 0)
    calcModeOverlaps: bool = False
    disableStepsizeWarning: bool = False
    disablePlotTimeWarning: bool = False
    disableDownsamplingWarning: bool = False
    
    # Solver parameters
    useAllCPUs: bool = False
    useGPU: bool = False
    Nx_main: int = 2
    Ny_main: int = 2
    dz_target: float = 1e-6
    padfactor: float = 1.5
    alpha: float = 3e14
    
    # Geometry parameters
    Lx_main: float = 1.0
    Ly_main: float = 1.0
    Lz: float = 1.0
    taperScaling: float = 1.0
    twistRate: float = 0.0
    bendingRoC: float = np.inf
    bendDirection: float = 0.0
    
    # Optical and material parameters
    lambda_: float = 1.0
    n_background: float = 1.0
    n_0: float = 1.0
    rho_e: float = 0.22
    
    # Computed properties
    def dx(self):
        return self.Lx_main / self.Nx_main
    
    def dy(self):
        return self.Ly_main / self.Ny_main
    
    def dz(self):
        return self.Lz / self.Nz()
    
    def Nx(self):
        targetLx = self.padfactor * self.Lx_main
        return round(targetLx / self.dx())
    
    def Ny(self):
        targetLy = self.padfactor * self.Ly_main
        return round(targetLy / self.dy())
    
    def Nz(self):
        return max(self.updates, round(self.Lz / self.dz_target))
    
    def Lx(self):
        return self.dx() * self.Nx()
    
    def Ly(self):
        return self.dy() * self.Ny()
    
    def x(self):
        return np.linspace(-self.Lx()/2, self.Lx()/2, self.Nx())
    
    def y(self):
        return np.linspace(-self.Ly()/2, self.Ly()/2, self.Ny())
    
    def finalizeVideo(self):
        print("Closing video handle (placeholder function)")
    
    # Placeholder methods for propagation and field manipulation
    def FD_BPM(self):
        pass
    
    def FFT_BPM(self):
        pass
    
    def initializeRIfromFunction(self, hFunc, *args):
        pass
    
    def initializeEfromFunction(self, hFunc, Eparameters):
        pass
    
    def offsetField(self, direction, distance):
        pass
    
    def tiltField(self, direction, angle):
        pass

