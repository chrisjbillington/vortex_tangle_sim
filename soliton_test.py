from __future__ import division, print_function
import numpy as np
import pylab as pl
from scipy.linalg import expm
from FEDVR import FiniteElements1D


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
N_1D, omega = get_number_and_trap(rho_max, R)  # 1D normalisation constant and harmonic trap frequency
                                               # corresponding to the desired maximum density and radius
mu = g*rho_max                              # The chemical potential

# Space:
x_min = -15e-6
x_max = 15e-6

# Finite elements:
N = 7
n_elements = 100
assert not (n_elements % 2) # Algorithm below relies on last element being odd numbered
elements = FiniteElements1D(N, n_elements, x_min, x_max)

# Which elements are odd numbered, and which are even?
element_numbers = np.array(range(n_elements))
even = (element_numbers % 2) == 0
odd = ~even
# Which are internal, which are on boundaries?
internal = (0 < element_numbers) & (element_numbers < n_elements - 1)
odd_internal = odd & internal
even_internal = even & internal

# Second derivative operators, each is an (N x N) array. Different at the left
# and right boundary elements in order to impose zero boundary conditions:
grad2_left, grad2, grad2_right = elements.second_derivative_operators()

# Density operator. Is diagonal and so is represented as a length N array
# containing its diagonals:
density_operator = elements.density_operator()

# The spatial points of the DVR basis functions, an (n_elements x N) array
x = elements.points

# The Harmonic trap at our gridpoints, (n_elements x N):
V = 0.5*m*omega**2*x**2

def plot(psi):
    x_interp, psi_interp = elements.interpolate_vector(psi, 100)
    mod2psi_exact = rho_max*(1 - x_interp**2/R**2)
    mod2psi_exact = np.clip(mod2psi_exact, 0, None)
    points, values = elements.get_values(psi)
    pl.plot(x_interp*1e6, mod2psi_exact, 'k-')
    pl.plot(x_interp*1e6, np.abs(psi_interp)**2, 'r-')
    # pl.plot(x_interp*1e6, np.abs(psi_interp.real)**2, 'g-')
    pl.plot(x_interp*1e6, (pl.angle(psi_interp)+pi)/(2*pi)*3e20, 'g-')
    # pl.plot(points*1e6, np.abs(values)**2*np.sign(values.real), 'ko')
    pl.grid(True)
    # for edge in elements.element_edges:
    #     pl.axvline(edge*1e6, linestyle='--', color='k')
    pl.axis([-15,15,0,3e20])
    pl.savefig('output.png')
    # pl.show()
    pl.clf()

def initial():
    dt = 1e-7
    t_final = 1

    # The potential energy evolution operator for half a timestep in imaginary
    # time, (n_elements x N). It is a diagonal operator, so the below array is
    # just the diagonals and is applied by multiplying psi elementwise:
    U_V_halfstep = np.exp(-1/hbar * V * dt/2)

    # The kinetic energy evolution oparators for half a timestep in imaginary
    # time, each is (N x N). Not diagonal, but the same in each element. We
    # need different operators for the first and last elements in order to
    # impose boundary conditions
    U_K_halfstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2) * dt/2)
    U_K_right_halfstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2_right) * dt/2)
    # The same as above but for a full timestep:
    U_K_left_fullstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2_left) * dt)
    U_K_fullstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2) * dt)

    def renormalise(psi):
        # imposing normalisation on the wavefunction:
        p = psi[:,:-1] # Don't double count edges, ok to exclude last edge (should be zero there)
        ncalc = np.vdot(p, p)
        print(ncalc)
        psi[:] *= np.sqrt(N_1D/ncalc)

    # The initial guess:
    def initial_guess(x):
        sigma = 0.5*R
        return N_1D/np.sqrt(2*pi*sigma**2)*np.exp(-x**2/(2*sigma**2))

    psi = elements.make_vector(initial_guess)
    renormalise(psi)

    # Guess for the chemical potential, will be updated thoughout the
    # simulation:
    mu = g*rho_max

    # The potential energy evolution operator for the first half timestep in
    # imaginary time. It is always the same as at the end of timesteps, so we
    # usually just re-use at the start of each loop. But this being the first
    # loop we need it now too.
    density = psi.conj()*density_operator*psi
    U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)

    i = 0
    t = 0
    import time
    start_time = time.time()
    while t < t_final:
        # Evolve for half a step with potential evolution operator:
        psi = U_V_halfstep*psi

        # Evolve odd elements for half a step with kinetic energy evolution operator:
        psi[odd_internal] = np.einsum('ij,nj->ni', U_K_halfstep, psi[odd_internal])
        psi[-1] = np.einsum('ij,j->i', U_K_right_halfstep, psi[-1])

        # Copy odd endpoints -> adjacent even endpoints:
        psi[even, -1] = psi[odd, 0]
        psi[even_internal, 0] = psi[odd_internal, -1]

        # Evolve even elements for a full step with kinetic energy evolution operator:
        psi[even_internal] = np.einsum('ij,nj->ni', U_K_fullstep, psi[even_internal])
        psi[0] = np.einsum('ij,j->i', U_K_left_fullstep, psi[0])

        # Copy even elements -> adjacent odd endpoints:
        psi[odd_internal, -1] = psi[even_internal, 0]
        psi[odd, 0] = psi[even, -1]

        # Evolve odd elements for half a step with kinetic energy evolution operator:
        psi[odd_internal] = np.einsum('ij,nj->ni', U_K_halfstep, psi[odd_internal])
        psi[-1] = np.einsum('ij,j->i', U_K_right_halfstep, psi[-1])

        # Renormalise:
        renormalise(psi)

        # Calculate nonlinear evolution operator for half a step
        density = psi.conj()*density_operator*psi
        U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)

        # Evolve for half a timestep with potential evolution operator:
        psi = U_V_halfstep*psi

        if not i % 10000:
            print((time.time() - start_time)/(i+1)*1e6, 'us per loop')
            # Renormalise:
            renormalise(psi)
            plot(psi)
            # print(i)

        i += 1
        t += dt


def evolution():
    dt = 2e-8
    t_final = 30e-3

    # The potential energy unitary evolution operator for half a timestep,
    # (n_elements x N). It is a diagonal operator, so the below array is just the
    # diagonals and is applied by multiplying psi elementwise:
    U_V_halfstep = np.exp(-1j/hbar * V * dt/2)

    # The kinetic energy unitary evolution oparators for half a timestep, each is
    # (N x N). Not diagonal, but the same in each element. We need different operators for
    # the first and last elements in order to impose boundary conditions
    U_K_left_halfstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2_left) * dt/2)
    U_K_halfstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2) * dt/2)
    U_K_right_halfstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2_right) * dt/2)
    # The same as above but for a full timestep:
    U_K_left_fullstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2_left) * dt)
    U_K_fullstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2) * dt)
    U_K_right_fullstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2_right) * dt)




if __name__ == '__main__':
    import bprofile
    initial()
