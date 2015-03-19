from __future__ import division, print_function
import time
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
mu_guess = g*rho_max                        # The estimated chemical potential of the groundstate

# Space:
x_min = -15e-6
x_max = 15e-6

# Finite elements:
N = 15
n_elements = 10
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

# Second derivative operator, (N x N):
grad2 = elements.second_derivative_operators()

# Density operator. Is diagonal and so is represented as a length N array
# containing its diagonals:
density_operator = elements.density_operator()

# The spatial points of the DVR basis functions, an (n_elements x N) array
x = elements.points

# The Harmonic trap at our gridpoints, (n_elements x N):
V = 0.5*m*omega**2*x**2

def plot(i, t, psi):
    x_plot = x.flatten()
    x_interp, values_interp = elements.interpolate_vector(psi, 100)

    mod2psi_tf = rho_max*(1 - x_plot**2/R**2)
    mod2psi_tf = np.clip(mod2psi_tf, 0, None)
    pl.plot(x_plot*1e6, mod2psi_tf, 'k-')
    pl.plot(x_plot*1e6, (pl.angle(psi.flatten())+pi)/(2*pi)*3e20, 'g-')
    pl.plot(x_interp*1e6, np.abs(values_interp)**2, 'b-')
    pl.grid(True)
    # for edge in elements.element_edges:
    #     pl.axvline(edge*1e6, linestyle='--', color='k')
    pl.axis([-15,15,0,3e20])
    pl.figtext(0.15, 0.125, r'$t=%.00f\,\mathrm{ms}$'%(t*1e3))
    pl.savefig('output.png')
    # pl.show()
    pl.clf()


def initial():

    def kinetic_evolution_operators(dt):
        # The kinetic energy evolution operator for half a timestep in
        # imaginary time, each is (N x N). Not diagonal, but the same in each
        # element.
        U_K_halfstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2) * dt/2)
        # The same as above but for a full timestep:
        U_K_fullstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2) * dt)

        return U_K_halfstep, U_K_fullstep

    def renormalise(psi):
        # imposing normalisation on the wavefunction:
        p = psi[:,:-1] # Don't double count edges
        ncalc = np.vdot(p, p)
        psi[:] *= np.sqrt(N_1D/ncalc)

        # # Impose a dark soliton:
        # psi[x<0] = -np.abs(psi[x < 0])
        # psi[x>=0] = np.abs(psi[x >= 0])

    def compute_mu(psi):

        # Kinetic energy operator:
        K = -hbar**2/(2*m)*grad2

        # Kinetic energy operator operating on psi:
        K_psi = np.einsum('ij,nj->ni', K, psi)
        K_psi[1:,0] += K_psi[:-1,-1]
        K_psi[:-1, -1] = K_psi[1:, 0]

        # Total norm:
        p = psi[:,:-1] # Don't double count edges
        ncalc = np.vdot(p, p).real

        # Total Hamaltonian:
        density = psi.conj()*density_operator*psi
        H_psi = K_psi + V + g * density * psi

        # Expectation value and uncertainty of Hamiltonian gives the
        # expectation value and uncertainty of the chemical potential:
        mu = np.vdot(p, H_psi[:,:-1]).real/ncalc
        mu2 = np.vdot(H_psi[:,:-1], H_psi[:,:-1]).real/ncalc
        u_mu = np.sqrt(mu2 - mu**2)

        return mu, u_mu/mu

    # The initial guess:
    def initial_guess(x):
        sigma = 0.5*R
        f = np.sqrt(N_1D/np.sqrt(2*pi*sigma**2)*np.exp(-x**2/(2*sigma**2)))
        # f *= (x/5e-6) # sensible initial guess for a dark soliton
        return f

    psi = elements.make_vector(initial_guess)
    renormalise(psi)

    # Estimate of chemical potential, will be updated throughout simulation:
    mu = mu_guess

    dt = 1e-7
    U_K_halfstep, U_K_fullstep = kinetic_evolution_operators(dt)

    # The potential energy evolution operator for the first half timestep in
    # imaginary time. It is always the same as at the end of timesteps, so we
    # usually just re-use at the start of each loop. But this being the first
    # loop we need it now too.
    density = psi.conj()*density_operator*psi
    U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)

    i = 0
    t = 0
    start_time = time.time()
    # How often, in timesteps, should we consider adjusting the timstep size?
    timestep_adjust_period = 1000
    while True:
        # Evolve for half a step with potential evolution operator:
        psi = U_V_halfstep*psi

        # Evolve odd elements for half a step with kinetic energy evolution operator:
        psi[odd] = np.einsum('ij,nj->ni', U_K_halfstep, psi[odd])

        # Copy odd endpoints -> adjacent even endpoints:
        psi[even, -1] = psi[odd, 0]
        psi[even_internal, 0] = psi[odd_internal, -1]
        psi[0, 0] = psi[-1, -1]

        # Evolve even elements for a full step with kinetic energy evolution operator:
        psi[even] = np.einsum('ij,nj->ni', U_K_fullstep, psi[even])

        # Copy even endpoints -> adjacent odd endpoints:
        psi[odd_internal, -1] = psi[even_internal, 0]
        psi[odd, 0] = psi[even, -1]
        psi[-1, -1] = psi[0, 0]

        # Evolve odd elements for half a step with kinetic energy evolution operator:
        psi[odd] = np.einsum('ij,nj->ni', U_K_halfstep, psi[odd])

        # Renormalise:
        renormalise(psi)

        # Calculate potential energy evolution operator for half a step
        density = psi.conj()*density_operator*psi
        U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)

        # Evolve for half a timestep with potential evolution operator:
        psi = U_V_halfstep*psi

        # The below three if statements execute on successive iterations.
        # They measure the effect of using a larger or smaller timestep,
        # and adjust it accordingly.
        if i % timestep_adjust_period == 0:
            # Assess whether larger or smaller timesteps lead to faster
            # convergence. First, record how fast energy is going down in one
            # step with the timestep what it is now:
            mu, convergence = compute_mu(psi)
            # Updating this because it depends on mu, which we just
            # recalculated:
            U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)

        elif i % timestep_adjust_period == 1:
            # print('testing timestep')
            # How fast did mu decrease when we left the timestep the same?
            previous_mu = mu
            mu, convergence = compute_mu(psi)
            mu_change_same = mu - previous_mu
            # Now make it bigger, see what happens:
            unchanged_timestep = dt
            dt = unchanged_timestep*2
            # Recompute evolution matrices:
            U_K_halfstep, U_K_fullstep = kinetic_evolution_operators(dt)
            U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)

        elif i % timestep_adjust_period == 2:
            # print('testing bigger timestep')
            # How fast did mu decrease when we made the timestep bigger?
            previous_mu = mu
            mu, convergence = compute_mu(psi)
            mu_change_bigger = mu - previous_mu
            # Now make it smaller, see what happens:
            dt = unchanged_timestep/2
            # Recompute evolution matrices:
            U_K_halfstep, U_K_fullstep = kinetic_evolution_operators(dt)
            U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)

        elif i % timestep_adjust_period == 3:
            # print('testing smaller timestep')
            # How fast did mu decrease when we made the timestep smaller?
            previous_mu = mu
            mu, convergence = compute_mu(psi)
            mu_change_smaller = mu - previous_mu

            # Which timestep gave us the biggest decrease in mu?
            best_change = min(mu_change_smaller, mu_change_same, mu_change_bigger)
            if best_change == mu_change_same:
                # leave timestep the same as before:
                dt = unchanged_timestep
                # print('leaving dt = ', dt)
            elif best_change == mu_change_smaller:
                # Smallest timestep is best timestep:
                dt = unchanged_timestep / 2
                print('setting dt = ', dt)
            elif best_change == mu_change_bigger:
                dt = unchanged_timestep
                # biggest timestep is best timestep
                # dt = unchanged_timestep * 2
                # print('setting dt = ', dt)

            # Recompute evolution matrices:
            U_K_halfstep, U_K_fullstep = kinetic_evolution_operators(dt)
            U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu) * dt/2)



        if not i % 10000:
            print((time.time() - start_time)/(i+1)*1e6, 'us per loop')
            renormalise(psi)
            plot(i, t, psi)
            mu, convergence = compute_mu(psi)
            print('convergence:', convergence)
            print('mu:', mu)

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
