from __future__ import division, print_function
import os
import time
import numpy as np
from scipy.linalg import expm
from FEDVR import FiniteElements2D

from PyQt4 import QtGui
from qtutils import inthread, inmain_decorator

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
x_min = -10e-6
x_max = 10e-6
y_min = -10e-6
y_max = 10e-6

# Finite elements:
Nx = 7
Ny = 7
n_elements_x = 32
n_elements_y = 32
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
    global image_item

    x_plot, y_plot, psi_interp = elements.interpolate_vector(psi, Nx, Ny)
    rho_plot = np.abs(psi_interp)**2

    if image_item is None:
        image_item = image_view.setImage(rho_plot)
    else:
        image_item.setData(rho_plot)



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


def renormalise(psi):
    # imposing normalisation on the wavefunction:
    p = psi[:, :, :-1, :-1] # Don't double count edges
    ncalc = np.vdot(p, p)
    psi[:] *= np.sqrt(N_2D/ncalc)


def initial():

    RELAXATION_PARAMETER = 1.7

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
    points_and_indices.append((edges, None, None))
    for j in range(1, Nx-1):
        for k in range(1, Ny-1):
            points_and_indices.append(((j_indices == j) & (k_indices == k), j, k))

    start_time = time.time()
    while True:
        # We operate on all elements at once, but one DVR basis function at a time.
        for points, j, k, in points_and_indices:
            if points is edges:
                # I lied about doing one basis function at a time.  We do all
                # the edge points simultaneously:
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
            else:
                # Internal DVR points we actually do do one at a time:
                Kx_psi[:, :, j, k] = np.einsum('j,xyj->xy', Kx[j, :],  psi[:, :, :, k])
                Ky_psi[:, :, j, k] = np.einsum('l,xyl->xy', Ky[k, :],  psi[:, :, j, :])

            # Total kinetic energy operator operating on psi at the DVR point(s):
            K_psi = Kx_psi[:, :, points] + Ky_psi[:, :, points]

            # Density at the DVR point(s):
            density = psi[:, :, points].conj() * density_operator[points] * psi[:, :, points]

            # Diagonals of the total Hamiltonian operator at the DVR point(s):
            H_diags = K_diags[points] + V[:, :, points] + g * density

            # Hamiltonian with diagonals subtracted off, operating on psi at the DVR point(s):
            H_hollow_psi = K_psi - K_diags[points] * psi[:, :, points]

            # The Gauss-Seidel prediction for the new psi at the DVR point(s):
            psi_new_GS = (mu * psi[:, :, points] - H_hollow_psi)/H_diags

            # Update psi at the DVR point(s) with overrelaxation:
            psi[:, :, points] += RELAXATION_PARAMETER * (psi_new_GS - psi[:, :, points])

        i += 1

        if not i % 100:
            mucalc, u_mucalc = compute_mu(psi)
            convergence = abs((mucalc - mu)/mu)
            print(i, mu, mucalc, convergence, 1e3*(time.time() - start_time)/i, 'msps')
            if convergence < 1e-13:
                plot(psi, show=False)
                break
        if not i % 10:
            plot(psi, show=False)

    return psi


def evolution(psi, t_final, dt=None, imaginary_time=False):

    global time_of_last_plot
    if dt is None:
        dx_max = np.diff(x[0, 0, :, 0]).max()
        dx_min = np.diff(x[0, 0, :, 0]).min()
        dy_max = np.diff(y[0, 0, 0, :]).max()
        dy_min = np.diff(y[0, 0, 0, :]).min()
        dt = max(dx_max, dy_max) * min(dx_min, dy_min) * m / (8 * pi * hbar)
        print('using dt = ', dt)
    n_initial = compute_number(psi)
    mu_initial, unc_mu_inintial = compute_mu(psi)

    if imaginary_time:
        # The kinetic energy unitary evolution oparators for half a timestep
        # in imaginary time, shapes (Nx, Nx) and (Ny, Ny). Not diagonal, but
        # the same in each element.
        U_Kx_halfstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2x) * dt/2)
        U_Ky_halfstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2y) * dt/2)
        # The same as above but for a full timestep:
        U_Kx_fullstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2x) * dt)
        U_Ky_fullstep = expm(-1/hbar * (-hbar**2/(2*m) * grad2y) * dt)
    else:
        # The kinetic energy unitary evolution oparators for half a timestep,
        # shapes (Nx, Nx) and (Ny, Ny). Not diagonal, but the same in each
        # element.
        U_Kx_halfstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2x) * dt/2)
        U_Ky_halfstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2y) * dt/2)
        # The same as above but for a full timestep:
        U_Kx_fullstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2x) * dt)
        U_Ky_fullstep = expm(-1j/hbar * (-hbar**2/(2*m) * grad2y) * dt)

    # The potential energy evolution operator for the first half timestep. It
    # is always the same as at the end of timesteps, so we usually just re-use
    # at the start of each loop. But this being the first loop we need it now
    # too.
    density = (psi.conj()*density_operator*psi).real
    if imaginary_time:
        U_V_halfstep = np.exp(-1/hbar * (g * density + V - mu_initial) * dt/2)
    else:
        U_V_halfstep = np.exp(-1j/hbar * (g * density + V - mu_initial) * dt/2)

    i = 0
    t = 0

    start_time = time.time()
    n_frames = 0
    while t < t_final:
        # Evolve for half a step with potential evolution operator:
        psi[:] = U_V_halfstep*psi


        # Evolve odd (in x direction) elements for half a step with x kinetic energy evolution operator:
        psi[odd_elements_x, :, :, :] = np.einsum('ij,xyjl->xyil', U_Kx_halfstep, psi[odd_elements_x, :, :, :])

        # Copy odd endpoints -> adjacent even endpoints in x direction:
        psi[even_elements_x, :, -1, :] = psi[odd_elements_x, :, 0, :]
        psi[even_internal_elements_x, :, 0, :] = psi[odd_internal_elements_x, :, -1, :]
        psi[0, :, 0, :] = psi[-1, :, -1, :] # periodic boundary conditions

        # Evolve even (in x direction) elements for a full step with x kinetic energy evolution operator:
        psi[even_elements_x, :, :, :] = np.einsum('ij,xyjl->xyil', U_Kx_fullstep, psi[even_elements_x, :, :, :])

        # Copy even endpoints -> adjacent odd endpoints in x direction:
        psi[odd_internal_elements_x, :, -1, :] = psi[even_internal_elements_x, :, 0, :]
        psi[odd_elements_x, :, 0, :] = psi[even_elements_x, :, -1, :]
        psi[-1, :, -1, :] = psi[0, :, 0, :] # periodic boundary conditions

        # Evolve odd (in x direction) elements for half a step with x kinetic energy evolution operator:
        psi[odd_elements_x, :, :, :] = np.einsum('ij,xyjl->xyil', U_Kx_halfstep, psi[odd_elements_x, :, :, :])

        # Copy odd endpoints -> adjacent even endpoints in x direction:
        psi[even_elements_x, :, -1, :] = psi[odd_elements_x, :, 0, :]
        psi[even_internal_elements_x, :, 0, :] = psi[odd_internal_elements_x, :, -1, :]
        psi[0, :, 0, :] = psi[-1, :, -1, :] # periodic boundary conditions



        # Evolve odd (in y direction) elements for half a step with y kinetic energy evolution operator:
        psi[:, odd_elements_y, :, :] = np.einsum('kl,xyjl->xyjk', U_Ky_halfstep, psi[:, odd_elements_y, :, :])

        # Copy odd endpoints -> adjacent even endpoints in x direction:
        psi[:, even_elements_y, :, -1] = psi[:, odd_elements_y, :, 0]
        psi[:, even_internal_elements_y, :, 0] = psi[:, odd_internal_elements_y, :, -1]
        psi[:, 0, :, 0] = psi[:, -1, :, -1] # periodic boundary conditions

        # Evolve even (in y direction) elements for a full step with y kinetic energy evolution operator:
        psi[:, even_elements_y, :, :] = np.einsum('kl,xyjl->xyjk', U_Ky_fullstep, psi[:, even_elements_y, :, :])

        # Copy even endpoints -> adjacent odd endpoints in x direction:
        psi[:, odd_internal_elements_y, :, -1] = psi[:, even_internal_elements_y, :, 0]
        psi[:, odd_elements_y, :, 0] = psi[:, even_elements_x, :, -1]
        psi[:, -1, :, -1] = psi[:, 0, :, 0] # periodic boundary conditions

        # Evolve odd (in y direction) elements for half a step with y kinetic energy evolution operator:
        psi[:, odd_elements_y, :, :] = np.einsum('kl,xyjl->xyjk', U_Ky_halfstep, psi[:, odd_elements_y, :, :])

        # Copy odd endpoints -> adjacent even endpoints in x direction:
        psi[:, even_elements_y, :, -1] = psi[:, odd_elements_y, :, 0]
        psi[:, even_internal_elements_y, :, 0] = psi[:, odd_internal_elements_y, :, -1]
        psi[:, 0, :, 0] = psi[:, -1, :, -1] # periodic boundary conditions


        # Calculate potential energy evolution operator for half a step
        if imaginary_time:
            renormalise(psi)
        density[:] = (psi.conj()*density_operator*psi).real
        if imaginary_time:
            U_V_halfstep[:] = np.exp(-1/hbar * (g * density + V - mu_initial) * dt/2)
        else:
            U_V_halfstep[:] = np.exp(-1j/hbar * (g * density + V - mu_initial) * dt/2)

        # Evolve for half a timestep with potential evolution operator:
        psi[:] = U_V_halfstep*psi

        if not i % 100:
            if imaginary_time:
                renormalise(psi)
            mucalc, u_mucalc = compute_mu(psi)
            ncalc = compute_number(psi)
            step_rate = (i+1)/(time.time() - start_time)
            frame_rate = n_frames/(time.time() - start_time)
            print(round(t*1e6), 'us',
                  round(np.log10(abs(ncalc/n_initial-1))),
                  round(np.log10(abs(mucalc/mu_initial-1))),
                  round(1e3/step_rate, 1), 'msps',
                  round(frame_rate, 1), 'fps')
        if (time.time() - time_of_last_plot) > 1/target_frame_rate:
            if imaginary_time:
                renormalise(psi)
            plot(psi, t, show=False)
            n_frames += 1
            time_of_last_plot = time.time()
        i += 1
        t += dt
    return psi

if __name__ == '__main__':

    import pyqtgraph as pg

    time_of_last_plot = time.time()
    target_frame_rate = 1

    qapplication = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.resize(800,800)
    image_view = pg.ImageView()
    image_item = None
    win.setCentralWidget(image_view)
    win.show()
    win.setWindowTitle('pyqtgraph example: ImageView')

    def dark_soliton(x, x_sol, rho, v):
        healing_length = hbar/np.sqrt(2*m*rho*g)
        v_sound = np.sqrt(rho*g/m)
        soliton_width = healing_length/np.sqrt(1 - v**2/v_sound**2)
        soliton_envelope = (1j*v/v_sound +
                            np.sqrt(1 + v**2/v_sound**2) *
                            np.tanh((x - x_sol)/(np.sqrt(2)*soliton_width)))
        return soliton_envelope

    def run_sims():
        # import lineprofiler
        # lineprofiler.setup()
        import cPickle
        if not os.path.exists('psi.pickle'):
            psi = initial()
            with open('psi.pickle', 'w') as f:
                cPickle.dump(psi, f)
        else:
            with open('psi.pickle') as f:
                psi = cPickle.load(f)

        # k = 2*pi*5/(10e-6)
        # # Give the condensate a kick:
        # psi *= np.exp(1j*k*x)

        # print a soliton onto the condensate:
        # density = (psi.conj()*density_operator*psi).real
        # x_soliton = 5e-6
        # rho_bg = density.max()#density[np.abs(x-x_soliton)==np.abs(x-x_soliton).min()]
        # v_soliton = 0#hbar*k/m
        # soliton_envelope = dark_soliton(x, x_soliton, rho_bg, v_soliton)
        # psi *= soliton_envelope

        # Scatter some vortices randomly about:
        for i in range(30):
            x_vortex = np.random.normal(0, scale=R)
            y_vortex = np.random.normal(0, scale=R)
            psi[:] *= np.exp(1j*np.arctan2(y - y_vortex, x - x_vortex))

        psi = evolution(psi, t_final=400e-6, imaginary_time=True)
        psi = evolution(psi, t_final=np.inf)

    # run_sims()
    inthread(run_sims)
    qapplication.exec_()


