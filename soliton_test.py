from __future__ import division
import numpy as np
import pylab as pl
from FEDVR import FiniteElements

def get_number_and_trap(rho_max, R):
    """Return the 1D normalisation constant N (units of atoms per unit area)
    and harmonic trap frequency omega that correspond to the Thomas-Fermi
    densiry profile with peak density rho_max and radial extent R"""
    N = 4/3*rho_max*R
    omega = np.sqrt(2*g*rho_max/(m*R**2))
    return N, omega

# Constants:
pi = np.pi
hbar = 1.054572e-34                         # Reduced Planck's constant
a_0  = 5.29177209e-11                       # Bohr radius
u    = 1.660539e-27                         # unified atomic mass unit
m  = 86.909180*u                            # 87Rb atomic mass
a  = 98.98*a_0                              # 87Rb |2,2> scattering length
g  = 4*pi*hbar**2*a/m                       # 87Rb self interaction constant
rho_max = 2.5e14*1e6                        # Desired max Thomas-Fermi density
R = 7.5e-6                                  # Desired Thomas-Fermi radius
N, omega = get_number_and_trap(rho_max, R)  # 1D normalisation constant and harmonic trap frequency
                                            # corresponding to the desired maximum density and radius
# Space:
x_min = -10e-6
x_max = 10e-6

# Dense grid for plotting:
x = np.linspace(x_min, x_max, 1000)

# Finite elements:
N = 7
n_elements = 20
elements = FiniteElements(N, n_elements, x_min, x_max)

def V(x):
    """The harmonic trap"""
    return 0.5*m*omega**2*x**2

def initial():
    """Imaginary time evolution for the initial state"""
    # The initial guess:
    def initial_guess(x):
        psi = rho_max*(1 - x**2/R**2)
        psi = np.clip(psi, 0, None)
        psi = np.sqrt(psi)
        psi *= (x<0) - 1*(x>=0)
        return psi

    vectors = elements.make_vectors(initial_guess)
    psi_interp, points, values = elements.interpolate_vectors(vectors, x)
    pl.plot(x*1e6, initial_guess(x), 'k-')
    pl.plot(x*1e6, psi_interp, 'r--')
    pl.plot(points*1e6, values, 'ko')
    for boundary in elements.boundaries:
        pl.axvline(boundary*1e6, linestyle='--', color='k')
    pl.grid(True)
    pl.show()

if __name__ == '__main__':
    initial()
