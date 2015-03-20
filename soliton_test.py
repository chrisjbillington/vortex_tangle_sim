from __future__ import division, print_function
import time
import numpy as np
import pylab as pl
from scipy.linalg import expm
from FEDVR import FiniteElements1D

import matplotlib
matplotlib.use("QT4Agg")
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from PyQt4 import QtCore, QtGui
from qtutils import inthread, inmain_decorator



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
n_elements = 50
assert not (n_elements % 2), "Odd-even split step method requires an even number of elements"
assert (N % 2), "Gauss Seidel checkerboard method requires odd number of basis functions per element"

elements = FiniteElements1D(N, n_elements, x_min, x_max)

# Which elements are odd numbered, and which are even?
element_numbers = np.array(range(n_elements))
even_elements = (element_numbers % 2) == 0
odd_elements = ~even_elements

# Which elements are internal, and which are on boundaries?
internal_elements = (0 < element_numbers) & (element_numbers < n_elements - 1)
odd_internal_elements = odd_elements & internal_elements
even_internal_elements = even_elements & internal_elements

# Which quadrature points are odd numbered, and which are even?
point_numbers = np.array(range(N))
even_points = (point_numbers % 2) == 0
odd_points = ~even_points

# Second derivative operator, (N x N):
grad2 = elements.second_derivative_operators()

# Density operator. Is diagonal and so is represented as a length N array
# containing its diagonals:
density_operator = elements.density_operator()

# The spatial points of the DVR basis functions, an (n_elements x N) array
x = elements.points

# The Harmonic trap at our gridpoints, (n_elements x N):
V = 0.5*m*omega**2*x**2

@inmain_decorator()
def plot(psi, t=None, show=False):
    x_plot, values = elements.get_values(psi)
    psi_tf = rho_max*(1 - x_plot**2/R**2)
    psi_tf = np.sqrt(np.clip(psi_tf, 0, None))

    if not figure.axes:
        pl.title('BEC in a trap')
        pl.xlabel('$x\ (\mu\mathrm{m})$')
        pl.ylabel('wavefunction real and imaginary parts')
        pl.axis([-15,15,-2e10,2e10])
    pl.gca().lines = []

    pl.plot(x_plot*1e6, psi_tf, 'k-')
    pl.plot(x_plot*1e6, values.real, 'b-')
    pl.plot(x_plot*1e6, values.imag, 'g-')
    pl.grid(True)


    figure.texts = []
    if t is not None:
        pl.figtext(0.15, 0.125, r'$t=%.00f\,\mathrm{ms}$'%(t*1e3))
    # pl.savefig('output.png')
    canvas.draw_idle()
    if show:
        pl.show()


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
    while True:
        # Alternately update even and odd points. This is the 'red black' Gauss Seidel method.
        if i % 2:
            points = odd_points
        else:
            points = even_points

        # Kinetic energy operator operating on psi at the relevant points:
        K_psi = np.einsum('ij,nj->ni', K[points, :], psi)
        # Edges of elements are even points. Add kinetic energy contributions
        # from either side of an edge if we are currently working on even
        # points:
        if points is even_points:
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
        if points is even_points:
            K_diags[0] *= 2
            K_diags[-1] *= 2
        H_diags = K_diags + V[:, points] + g * density
        H_hollow_psi = K_psi - K_diags*psi[:, points]

        # update the relevant points of psi
        psi[:, points] = (mu*psi[:, points] - H_hollow_psi)/H_diags

        i += 1

        if not i % 1000:
            previous_mucalc = mucalc
            mucalc, convergence = compute_mu(psi)
            print(i, mu, repr(mucalc), convergence)
            if abs(mucalc - previous_mucalc) < 1e-14:
                plot(psi, show=False)
                break
        if not i % 10000:
            plot(psi, show=False)

    return psi


def evolution(psi):

    # Give the condensate a kick:
    k = 2*pi*5/(10e-6)
    psi *= np.exp(1j*k*x)


    dx_max = np.diff(x[0,:]).max()
    dx_min = np.diff(x[0,:]).min()
    dt = dx_max*dx_min*m/(4*pi*hbar)
    t_final = 100e-3

    n_initial = compute_number(psi)

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
    density = psi.conj()*density_operator*psi
    U_V_halfstep = np.exp(-1j/hbar * (g * density + V - mu) * dt/2)

    i = 0
    t = 0
    while t < t_final:
        # Evolve for half a step with potential evolution operator:
        psi = U_V_halfstep*psi

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

        # Copy odd endpoints -> adjacent even endpoints:
        psi[even_elements, -1] = psi[odd_elements, 0]
        psi[even_internal_elements, 0] = psi[odd_internal_elements, -1]
        psi[0, 0] = psi[-1, -1]

        # Calculate potential energy evolution operator for half a step
        density = psi.conj()*density_operator*psi
        U_V_halfstep = np.exp(-1j/hbar * (g * density + V - mu) * dt/2)

        # Evolve for half a timestep with potential evolution operator:
        psi = U_V_halfstep*psi

        if not i % 1000:
            # print(i, t*1e3, 'ms')
            mucalc, u_mucalc = compute_mu(psi)
            ncalc = compute_number(psi)
            print(round(t*1e3), 'ms', ncalc/n_initial)
        if not i % 50:
            plot(psi, t, show=False)
        i += 1
        t += dt

if __name__ == '__main__':

    qapplication = QtGui.QApplication([])

    figure = pl.figure()
    canvas = FigureCanvas(figure)
    window = QtGui.QWidget()
    layout = QtGui.QVBoxLayout(window)
    navigation_toolbar = NavigationToolbar(canvas, window)
    layout.addWidget(navigation_toolbar)
    layout.addWidget(canvas)
    window.show()

    def run_sims():
        psi = initial()
        psi = evolution(psi)

    def start(*args):
        inthread(run_sims)

    timer = QtCore.QTimer.singleShot(0, start)

    qapplication.exec_()


