from __future__ import division, print_function
import time
import numpy as np
from scipy.linalg import expm
from FEDVR import FiniteElements2D

from PyQt4 import QtGui
from qtutils import inthread, inmain_decorator
import pyqtgraph.opengl as gl

from IPython import embed as DEBUG

def get_number_and_trap(rho_max, R):
    """Return the 1D normalisation constant N (units of atoms per unit area)
    and harmonic trap frequency omega that correspond to the Thomas-Fermi
    densiry profile with peak density rho_max and radial extent R"""
    N_2D = pi*rho_max*R**2/2
    omega = np.sqrt(2*g*rho_max/(m*R**2))
    return N_2D, omega

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
N_2D, omega = get_number_and_trap(rho_max, R)  # 2D normalisation constant and harmonic trap frequency
                                               # corresponding to the desired maximum density and radius
mu = g*rho_max                              # Chemical potential of the groundstate

# Space:
x_min = -15e-6
x_max = 15e-6
y_min = -15e-6
y_max = 15e-6

# Finite elements:
Nx = 7
Ny = 7
n_elements_x = 64
n_elements_y = 64
assert not (n_elements_x % 2), "Odd-even split step method requires an even number of elements"
assert not (n_elements_y % 2), "Odd-even split step method requires an even number of elements"

elements = FiniteElements2D(Nx, Ny, n_elements_x, n_elements_y, x_min, x_max, y_min, y_max)

# Which elements are odd numbered, and which are even?
element_numbers_x = np.array(range(n_elements_x))
element_numbers_y = np.array(range(n_elements_y))
even_elements_x = (element_numbers_x % 2) == 0
odd_elements_x = ~even_elements_x
even_elements_y = (element_numbers_y % 2) == 0
odd_elements_y = ~even_elements_y

# Which elements are internal, and which are on boundaries?
internal_elements_x = (0 < element_numbers_x) & (element_numbers_x < n_elements_x - 1)
internal_elements_y = (0 < element_numbers_y) & (element_numbers_y < n_elements_y - 1)
odd_internal_elements_x = odd_elements_x & internal_elements_x
even_internal_elements_x = even_elements_x & internal_elements_x
odd_internal_elements_y = odd_elements_y & internal_elements_y
even_internal_elements_y = even_elements_y & internal_elements_y


# Second derivative operators, each (N x N):
grad2x, grad2y = elements.second_derivative_operators()

# Density operator. Is diagonal and so is represented as an (Nx x Ny) array
# containing its diagonals:
density_operator = elements.density_operator()

# The x spatial points of the DVR basis functions, an (n_elements_x x Nx) array:
x = elements.points_X
# The y spatial points of the DVR basis functions, an (n_elements_y x Ny) array:
y = elements.points_Y

# The Harmonic trap at our gridpoints, (n_elements x N):
V = 0.5*m*omega**2*(x**2 + y**2)

@inmain_decorator()
def plot(psi, t=None, show=False):
    global plot_item

    x_plot, y_plot, psi_interp = elements.get_values(psi)
    rho_plot = np.abs(psi_interp)**2

    if plot_item is None:
        plot_item = gl.GLSurfacePlotItem(x=x_plot*1e6, y=y_plot*1e6,
                                         z=10*rho_plot/rho_max, shader='normalColor')
        plot_window.addItem(plot_item)
        plot_window.show()
    else:
        plot_item.setData(x=x_plot*1e6, y=y_plot*1e6, z=10*rho_plot/rho_max)


def compute_mu(psi):

    # Kinetic energy operator:
    Kx = -hbar**2/(2*m)*grad2x
    Ky = -hbar**2/(2*m)*grad2y

    # x kinetic energy operator operating on psi.
    Kx_psi = np.einsum('ij,xyjl->xyil', Kx,  psi)

    # Add contributions from left to right across edges:
    Kx_psi[1:, :, 0, :] += Kx_psi[:-1, :, -1, :]
    Kx_psi[0, :, 0, :] += Kx_psi[-1, :, -1, :] # Periodic boundary conditions

    # Copy summed values back from right to left across edges:
    Kx_psi[:-1, :, -1, :] = Kx_psi[1:, :, 0, :]
    Kx_psi[-1, :, -1, :] = Kx_psi[0, :, 0, :]  # Periodic boundary conditions

    # y kinetic energy operator operating on psi.
    Ky_psi = np.einsum('kl,xyjl->xyjk', Ky,  psi)

    # Add contributions from top to bottom across edges:
    Ky_psi[:, 1:, :, 0] += Ky_psi[:, :-1, :, -1]
    Ky_psi[:, 0, :, 0] += Ky_psi[:, -1, :, -1] # Periodic boundary conditions

    # Copy summed values back from bottom to top across edges:
    Ky_psi[:, :-1, :, -1] = Ky_psi[:, 1:, :, 0]
    Ky_psi[:, -1, :, -1] = Ky_psi[:, 0, :, 0] # Periodic boundary conditions

    # Total kinetic energy operator operating on psi:
    K_psi = Kx_psi + Ky_psi

    # Total norm:
    p = psi[:, :, :-1, :-1] # Don't double count edges
    ncalc = np.vdot(p, p).real

    # Total Hamaltonian:
    density = psi.conj()*density_operator*psi
    H_psi = K_psi + (V + g * density) * psi

    # Expectation value and uncertainty of Hamiltonian gives the
    # expectation value and uncertainty of the chemical potential:
    mu = np.vdot(p, H_psi[:, :, :-1, :-1]).real/ncalc
    mu2 = np.vdot(H_psi[:, :, :-1, :-1], H_psi[:, :, :-1, :-1]).real/ncalc
    var_mu = mu2 - mu**2
    if var_mu < 0:
        u_mu = 0
    else:
        u_mu = np.sqrt(var_mu)
    return mu, u_mu/mu


def compute_number(psi):
    # Total norm:
    p = psi[:, :, :-1, :-1] # Don't double count edges
    ncalc = np.vdot(p, p).real
    return ncalc


def initial():

    RELAXATION_PARAMETER = 1.7

    def renormalise(psi):
        # imposing normalisation on the wavefunction:
        p = psi[:, :, :-1, :-1] # Don't double count edges
        ncalc = np.vdot(p, p)
        psi[:] *= np.sqrt(N_2D/ncalc)


    # The initial guess:
    def initial_guess(x, y):
        sigma_x = 0.5*R
        sigma_y = 0.5*R
        f = np.sqrt(np.exp(-x**2/(2*sigma_x**2) - y**2/(2*sigma_y**2)))
        return f

    psi = elements.make_vector(initial_guess)
    renormalise(psi)
    # plot(psi)

    # Kinetic energy operators:
    Kx = -hbar**2/(2*m)*grad2x
    Ky = -hbar**2/(2*m)*grad2y

    # Diagonals of the total kinetic energy operator, shape (Nx, Ny):
    Kx_diags = Kx.diagonal().copy()
    Kx_diags[0] *= 2
    Kx_diags[-1] *= 2
    Ky_diags = Ky.diagonal().copy()
    Ky_diags[0] *= 2
    Ky_diags[-1] *= 2
    K_diags = Kx_diags[:, np.newaxis] + Ky_diags[np.newaxis, :]

    i = 0
    mucalc, convergence = compute_mu(psi)

    # Arrays for storing the x and y kinetic energy terms,
    # which are needed by neighboring points
    Kx_psi = np.einsum('ij,xyjl->xyil', Kx,  psi)
    Ky_psi = np.einsum('kl,xyjl->xyjk', Ky,  psi)

    j_indices = np.array(range(Nx))[:, np.newaxis]*np.ones(Ny)[np.newaxis, :]
    k_indices = np.array(range(Ny))[np.newaxis, :]*np.ones(Ny)[:, np.newaxis]
    edges = (j_indices == 0) | (k_indices == 0) | (j_indices == Nx - 1) | (k_indices == Ny - 1)

    # Each loop we first update the edges of each element, then we loop
    # over the internal basis points. Here we just create the list of
    # indices we iterate over:
    points_and_indices = []
    points_and_indices.append((edges, 0, 0))
    for j in range(1, Nx-1):
        for k in range(1, Ny-1):
            points_and_indices.append(((j, k), j, k))
    start_time = time.time()
    while True:
        # We operate on all elements at once, but one DVR basis function at a time.
        # However, we do all edge points simultaneously:
        Kx_psi[:, :, (0, -1), :] = np.einsum('ij,xyjl->xyil', Kx[(0, -1), :],  psi)
        Kx_psi[:, :, :, (0, -1)] = np.einsum('ij,xyjl->xyil', Kx,  psi[:, :, :, (0, -1)])
        Ky_psi[:, :, (0, -1), :] = np.einsum('kl,xyjl->xyjk', Ky,  psi[:, :, (0, -1), :])
        Ky_psi[:, :, :, (0, -1)] = np.einsum('kl,xyjl->xyjk', Ky[(0, -1), :],  psi)

        # Add contributions from left to right across edges:
        Kx_psi[1:, :, 0, :] += Kx_psi[:-1, :, -1, :]
        Kx_psi[0, :, 0, :] += Kx_psi[-1, :, -1, :] # Periodic boundary conditions

        # Copy summed values back from right to left across edges:
        Kx_psi[:-1, :, -1, :] = Kx_psi[1:, :, 0, :]
        Kx_psi[-1, :, -1, :] = Kx_psi[0, :, 0, :]  # Periodic boundary conditions

        # Add contributions from top to bottom across edges:
        Ky_psi[:, 1:, :, 0] += Ky_psi[:, :-1, :, -1]
        Ky_psi[:, 0, :, 0] += Ky_psi[:, -1, :, -1] # Periodic boundary conditions

        # Copy summed values back from bottom to top across edges:
        Ky_psi[:, :-1, :, -1] = Ky_psi[:, 1:, :, 0]
        Ky_psi[:, -1, :, -1] = Ky_psi[:, 0, :, 0] # Periodic boundary conditions

        # Total kinetic energy operator operating on psi:
        K_psi = Kx_psi[:, :, edges] + Ky_psi[:, :, edges]

        density = psi[:, :, edges].conj() * density_operator[edges] * psi[:, :, edges]

        # Diagonals of the total Hamiltonian operator at this DVR point:
        H_diags = K_diags[edges] + V[:, :, edges] + g * density
        # Hamiltonian with diagonals subtracted off, operating on psi:
        H_hollow_psi = K_psi - K_diags[edges] * psi[:, :, edges]

        # The Gauss-Seidel prediction for the new psi:
        psi_new_GS = (mu * psi[:, :, edges] - H_hollow_psi)/H_diags

        # update the relevant points of psi
        psi[:, :, edges] += RELAXATION_PARAMETER * (psi_new_GS - psi[:, :, edges])

        for j in range(1, Nx-1):
            for k in range(1, Ny-1):
                Kx_psi[:, :, j, k] = np.einsum('j,xyj->xy', Kx[j, :],  psi[:, :, :, k])
                Ky_psi[:, :, j, k] = np.einsum('l,xyl->xy', Ky[k, :],  psi[:, :, j, :])

                # The kinetic energy term at this DVR point:
                K_psi = Kx_psi[:, :, j, k] + Ky_psi[:, :, j, k]

                # Particle density at this DVR point:
                density = psi[:, :, j, k].conj() * density_operator[j, k] * psi[:, :, j, k]

                # Diagonals of the total Hamiltonian operator at this DVR point:
                H_diags = K_diags[j, k] + V[:, :, j, k] + g * density
                # Hamiltonian with diagonals subtracted off, operating on psi:
                H_hollow_psi = K_psi - K_diags[j, k] * psi[:, :, j, k]

                # The Gauss-Seidel prediction for the new psi:
                psi_new_GS = (mu * psi[:, :, j, k] - H_hollow_psi)/H_diags

                # update the relevant points of psi
                psi[:, :, j, k] += RELAXATION_PARAMETER * (psi_new_GS - psi[:, :, j, k])

        i += 1

        if not i % 100:
            mucalc, u_mucalc = compute_mu(psi)
            convergence = abs((mucalc - mu)/mu)
            print(i, mu, mucalc, convergence, 1e3*(time.time() - start_time)/i, 'msps')
            if convergence < 1e-13:
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

    qapplication = QtGui.QApplication([])
    plot_window = gl.GLViewWidget()
    plot_window.setCameraPosition(distance=50)
    plot_item = None

    def run_sims():
        # import lineprofiler
        # lineprofiler.setup()
        psi = initial()

        # k = 2*pi*5/(10e-6)
        # # Give the condensate a kick:
        # # psi *= np.exp(1j*k*x)

        # # print a soliton onto the condensate:
        # density = (psi.conj()*density_operator*psi).real
        # x_soliton = 5e-6
        # rho_bg = density[np.abs(x-x_soliton)==np.abs(x-x_soliton).min()]
        # v_soliton = 0#hbar*k/m
        # soliton_envelope = dark_soliton(x, x_soliton, rho_bg, v_soliton)
        # psi *= soliton_envelope

        # # psi = imaginary_evolution(psi)
        # psi = evolution(psi)

    run_sims()
    # inthread(run_sims)
    qapplication.exec_()


