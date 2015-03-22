from __future__ import division, print_function
import time
import numpy as np
import pylab as pl
from scipy.linalg import expm
from FEDVR import FiniteElements1D

from PyQt4 import QtCore, QtGui
from qtutils import inthread, inmain_decorator

import pyqtgraph as pg

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
mu = g*rho_max                              # Chemical potential of the groundstate

# Space:
x_min = -15e-6
x_max = 15e-6

# Finite elements:
N = 7
n_elements = 64
assert not (n_elements % 2), "Odd-even split step method requires an even number of elements"

elements = FiniteElements1D(N, n_elements, x_min, x_max)

# Which elements are odd numbered, and which are even?
element_numbers = np.array(range(n_elements))
even_elements = (element_numbers % 2) == 0
odd_elements = ~even_elements

# Which elements are internal, and which are on boundaries?
internal_elements = (0 < element_numbers) & (element_numbers < n_elements - 1)
odd_internal_elements = odd_elements & internal_elements
even_internal_elements = even_elements & internal_elements

# Second derivative operator, (N x N):
grad2 = elements.second_derivative_operator()

# Density operator. Is diagonal and so is represented as a length N array
# containing its diagonals:
density_operator = elements.density_operator()

# The spatial points of the DVR basis functions, an (n_elements x N) array
x = elements.points

# The Harmonic trap at our gridpoints, (n_elements x N):
V = 0.5*m*omega**2*x**2

def dark_soliton(x, x_sol, rho, v):
    healing_length = hbar/np.sqrt(2*m*rho*g)
    v_sound = np.sqrt(rho*g/m)
    soliton_width = healing_length/np.sqrt(1 - v**2/v_sound**2)
    soliton_envelope = (1j*v/v_sound +
                        np.sqrt(1 + v**2/v_sound**2) *
                        np.tanh((x - x_sol)/(np.sqrt(2)*soliton_width)))
    return soliton_envelope

@inmain_decorator()
def plot(psi, t=None, show=False):
    global abs_curve, pot_curve, tf_curve

    x_plot, values = elements.get_values(psi)
    V_plot = V.flatten()/g

    if abs_curve is None:
        psi_tf = rho_max*(1 - x_plot**2/R**2)
        psi_tf = np.clip(psi_tf, 0, None)
        tf_curve = plot_window.plot(x_plot*1e6, np.abs(values)**2, pen='y')
        abs_curve = plot_window.plot(x_plot*1e6, np.abs(values)**2 + V_plot, pen='w')
        pot_curve = plot_window.plot(x_plot*1e6, V_plot, pen='g')
        plot_window.showGrid(True, True)
        plot_window.setXRange(-15, 15)
        plot_window.setYRange(0, 6e20)
    else:
        abs_curve.setData(x_plot*1e6, np.abs(values)**2 + V_plot, pen='w')
        tf_curve.setData(x_plot*1e6, np.abs(values)**2, pen='y')


def compute_mu(psi):

    # Kinetic energy operator:
    K = -hbar**2/(2*m)*grad2

    # Kinetic energy operator operating on psi:
    K_psi = np.einsum('ij,nj->ni', K, psi)
    K_psi[1:,0] += K_psi[:-1,-1]
    K_psi[:-1, -1] = K_psi[1:, 0]
    K_psi[0,0] += K_psi[-1,-1]
    K_psi[-1, -1] = K_psi[0, 0]

    # Total norm:
    p = psi[:,:-1] # Don't double count edges
    ncalc = np.vdot(p, p).real

    # Total Hamaltonian:
    density = psi.conj()*density_operator*psi
    H_psi = K_psi + (V + g * density) * psi

    # Expectation value and uncertainty of Hamiltonian gives the
    # expectation value and uncertainty of the chemical potential:
    mu = np.vdot(p, H_psi[:,:-1]).real/ncalc
    mu2 = np.vdot(H_psi[:,:-1], H_psi[:,:-1]).real/ncalc
    var_mu = mu2 - mu**2
    if var_mu < 0:
        u_mu = 0
    else:
        u_mu = np.sqrt(var_mu)
    return mu, u_mu/mu


def compute_number(psi):
    # Total norm:
    p = psi[:,:-1] # Don't double count edges
    ncalc = np.vdot(p, p).real
    return ncalc


def initial():

    RELAXATION_PARAMETER = 1.4

    def renormalise(psi):
        # imposing normalisation on the wavefunction:
        p = psi[:,:-1] # Don't double count edges
        ncalc = np.vdot(p, p)
        psi[:] *= np.sqrt(N_1D/ncalc)

        # # Impose a dark soliton:
        # psi[x<0] = -np.abs(psi[x < 0])
        # psi[x>=0] = np.abs(psi[x >= 0])

    # The initial guess:
    def initial_guess(x):
        sigma = 0.5*R
        f = np.sqrt(N_1D/np.sqrt(2*pi*sigma**2)*np.exp(-x**2/(2*sigma**2)))
        # f *= (x/5e-6) # sensible initial guess for a dark soliton
        return f

    psi = elements.make_vector(initial_guess)
    renormalise(psi)

    # Kinetic energy operator:
    K = -hbar**2/(2*m)*grad2

    i = 0
    mucalc, convergence = compute_mu(psi)

    # An array for selecting which points we are operating on. For the SOR
    # method, we update all elements simultaneously, one basis function at a
    # time, except for the endpoints which we do simultaneously.
    pointslist = np.zeros((N-1, N), dtype=bool)
    for j in range(N-1):
        pointslist[j,j] = True
    pointslist[0,N-1] = True

    while True:
        for j in range(N-1):
            points = pointslist[j]

            # Kinetic energy operator operating on psi at the relevant points:
            K_psi = np.einsum('ij,nj->ni', K[points, :], psi)
            # Add kinetic energy contributions from either side of an edge if we
            # are currently working on endpoints
            if j == 0:
                K_psi[1:,0] += K_psi[:-1,-1]
                K_psi[:-1, -1] = K_psi[1:, 0]
                K_psi[0,0] += K_psi[-1,-1]
                K_psi[-1, -1] = K_psi[0, 0]

            # Particle density at the relevant points:
            density = psi[:, points].conj()*density_operator[points]*psi[:, points]
            # density = psi.conj()*density_operator*psi

            # Diagonals of the total Hamiltonian operator at the relevant points.
            # Shape (n_elements x N/2), where N/2 is rounded up to an integer if
            # we're doing the even points and rounded down if we're doing the odd
            # points.
            K_diags = K[points, points].copy()
            if j == 0:
                K_diags[0] *= 2
                K_diags[-1] *= 2
            H_diags = K_diags + V[:, points] + g * density
            H_hollow_psi = K_psi - K_diags*psi[:, points]

            # The Gauss-Seidel prediction for the new psi:
            psi_new_GS = (mu*psi[:, points] - H_hollow_psi)/H_diags

            # update the relevant points of psi
            psi[:, points] += RELAXATION_PARAMETER*(psi_new_GS - psi[:, points])

        i += 1

        if not i % 1000:
            mucalc, u_mucalc = compute_mu(psi)
            convergence = abs((mucalc - mu)/mu)
            print(i, convergence)
            if convergence < 1e-14:
                plot(psi, show=False)
                break
        if not i % 100:
            plot(psi, show=False)

    return psi


def evolution(psi):

    global time_of_last_plot

    dx_max = np.diff(x[0,:]).max()
    dx_min = np.diff(x[0,:]).min()
    dt = dx_max*dx_min*m/(8*pi*hbar)
    t_final = 1e-3


    dx_max = np.diff(x[0,:]).max()
    dx_min = np.diff(x[0,:]).min()
    dt = dx_max*dx_min*m/(8*pi*hbar)
    t_final = 100e-3

    n_initial = compute_number(psi)
    mu_initial, unc_mu_inintial = compute_mu(psi)

    # The kinetic energy unitary evolution oparators for half a timestep, each is
    # (N x N). Not diagonal, but the same in each element. We need different operators for
    # the first and last elements in order to impose boundary conditions
    U_K_halfstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2) * dt/2)
    # The same as above but for a full timestep:
    U_K_fullstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2) * dt)

    # The potential energy evolution operator for the first half timestep. It
    # is always the same as at the end of timesteps, so we usually just re-use
    # at the start of each loop. But this being the first loop we need it now
    # too.
    density = (psi.conj()*density_operator*psi).real
    U_V_halfstep = np.exp(-1j/hbar * (g * density + V - mu) * dt/2)

    i = 0
    t = 0

    start_time = time.time()
    n_frames = 0
    while True: # t < t_final:
        # Evolve for half a step with potential evolution operator:
        psi[:] = U_V_halfstep*psi

        # Evolve odd elements for half a step with kinetic energy evolution operator:
        psi[odd_elements] = np.einsum('ij,nj->ni', U_K_halfstep, psi[odd_elements])

        # Copy odd endpoints -> adjacent even endpoints:
        psi[even_elements, -1] = psi[odd_elements, 0]
        psi[even_internal_elements, 0] = psi[odd_internal_elements, -1]
        psi[0, 0] = psi[-1, -1]

        # Evolve even elements for a full step with kinetic energy evolution operator:
        psi[even_elements] = np.einsum('ij,nj->ni', U_K_fullstep, psi[even_elements])

        # Copy even endpoints -> adjacent odd endpoints:
        psi[odd_internal_elements, -1] = psi[even_internal_elements, 0]
        psi[odd_elements, 0] = psi[even_elements, -1]
        psi[-1, -1] = psi[0, 0]

        # Evolve odd elements for half a step with kinetic energy evolution operator:
        psi[odd_elements] = np.einsum('ij,nj->ni', U_K_halfstep, psi[odd_elements])

        # Calculate potential energy evolution operator for half a step
        density[:] = (psi.conj()*density_operator*psi).real
        U_V_halfstep[:] = np.exp(-1j/hbar * (g * density + V - mu) * dt/2)

        # Evolve for half a timestep with potential evolution operator:
        psi[:] = U_V_halfstep*psi

        if not i % 1000:
            # print(i, t*1e3, 'ms')
            mucalc, u_mucalc = compute_mu(psi)
            ncalc = compute_number(psi)
            step_rate = (i+1)/(time.time() - start_time)
            frame_rate = n_frames/(time.time() - start_time)
            print(round(t*1e3), 'ms',
                  round(np.log10(abs(ncalc/n_initial-1))),
                  round(np.log10(abs(mucalc/mu_initial-1))),
                  round(1e6/step_rate, 1), 'usps',
                  round(frame_rate, 1), 'fps')
        if (time.time() - time_of_last_plot) > 1/target_frame_rate:

            # Copy odd endpoints -> adjacent even endpoints:
            psi[even_elements, -1] = psi[odd_elements, 0]
            psi[even_internal_elements, 0] = psi[odd_internal_elements, -1]
            psi[0, 0] = psi[-1, -1]

            plot(psi, t, show=False)
            n_frames += 1
            time_of_last_plot = time.time()
        i += 1
        t += dt
    return psi

if __name__ == '__main__':

    time_of_last_plot = time.time()
    target_frame_rate = 10

    plot_window = pg.PlotWindow()
    abs_curve = None
    pot_curve = None
    tf_curve = None

    def run_sims():
        psi = initial()

        k = 2*pi*5/(10e-6)
        # Give the condensate a kick:
        # psi *= np.exp(1j*k*x)

        # print a soliton onto the condensate:
        density = (psi.conj()*density_operator*psi).real
        x_soliton = 5e-6
        rho_bg = density[np.abs(x-x_soliton)==np.abs(x-x_soliton).min()]
        v_soliton = 0#hbar*k/m
        soliton_envelope = dark_soliton(x, x_soliton, rho_bg, v_soliton)
        psi *= soliton_envelope

        # psi = imaginary_evolution(psi)
        psi = evolution(psi)

    # run_sims()
    inthread(run_sims)
    QtGui.QApplication.instance().exec_()


