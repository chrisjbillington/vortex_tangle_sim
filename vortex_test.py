from __future__ import division, print_function
import os

import numpy as np
from PyQt4 import QtCore, QtGui
from qtutils import inthread, inmain_decorator

from BEC2D import Simulator2D

# Constants:
pi = np.pi
hbar = 1.054571726e-34
a_0  = 5.29177209e-11                       # Bohr radius
u    = 1.660539e-27                         # unified atomic mass unit
m  = 86.909180*u                            # 87Rb atomic mass
a  = 98.98*a_0                              # 87Rb |2,2> scattering length
g  = 4*pi*hbar**2*a/m                       # 87Rb self interaction constant
rho_max = 2.5e14*1e6                        # Desired maximum density
R = 7.5e-6                                  # Desired Thomas-Fermi radius
omega = np.sqrt(2*g*rho_max/(m*R**2))       # Trap frequency  corresponding to desired density and radius
mu = g*rho_max                              # Desired chemical potential of the groundstate
N_2D = pi*rho_max*R**2/2                    # Thomas Fermi estimate of atom number for given chemical potential.

# Total spatial region over all MPI processes:
x_min_global = -10e-6
x_max_global = 10e-6
y_min_global = -10e-6
y_max_global = 10e-6

# Finite elements:
n_elements_x_global = 8
n_elements_y_global = 8

# Number of DVR basis functions per element:
Nx = 7
Ny = 7

simulator = Simulator2D(x_min_global, x_max_global, y_min_global, y_max_global, Nx, Ny,
                        n_elements_x_global, n_elements_y_global, output_file = 'vortex_test_rk4.h5')
x = simulator.x
y = simulator.y

# Kinetic energy operators:
Kx = -hbar**2/(2*m) * simulator.grad2x
Ky = -hbar**2/(2*m) * simulator.grad2y

# The Harmonic trap at our gridpoints, (n_elements_x, n_elements_y, Nx, Ny):
V = 0.5*m*omega**2*(x**2 + y**2)

def H(t, psi, x_elements, y_elements, x_points, y_points):
    psi[:] = 0
    psi[x_elements, y_elements, x_points, y_points] = 1
    plot(psi, None)
    import time
    time.sleep(5)
    Kx_psi = np.einsum('ij,xyjl->xyil', Kx[x_points, :], psi[x_elements, y_elements, :, y_points])
    Ky_psi = np.einsum('kl,xyjl->xyjk', Ky[y_points, :], psi[x_elements, y_elements, x_points, :])
    K_psi = Kx_psi + Ky_psi
    U = V[x_elements, y_elements, x_points, y_points]
    U_nonlinear = g * simulator.density_operator[x_points, y_points]
    return K_psi, U, U_nonlinear

def K_diags(x, y):
    """Return the diagonals of the nondiagonal part of the Hamiltonian. Must
    have the same dimensionality as psi, so we add dimensions of size one for
    n_elements_x and n_elements_y. If we had rotation terms or something then
    our returned array would actually vary over those dimensions."""
    return (Kx.diagonal()[np.newaxis, np.newaxis, :, np.newaxis] +
            Ky.diagonal()[np.newaxis, np.newaxis, np.newaxis, :])

@inmain_decorator()
def plot(psi, output_log):
    if SHOW_PLOT:
        import matplotlib
        global image_item
        # x_plot, y_plot, psi_interp = simulator.elements.interpolate_vector(psi, Nx, Ny)
        psi_interp = psi.transpose(0,2,1,3).reshape((n_elements_x_global*Nx, n_elements_y_global*Ny))
        psi_interp[psi_interp > 1] = 1
        rho = np.abs(psi_interp)**2
        phase = np.angle(psi_interp)

        hsl = np.zeros(psi_interp.shape + (3,))
        hsl[:, :, 2] = rho/rho.max()
        hsl[:, :, 0] = np.array((phase + pi)/(2*pi))
        hsl[:, :, 1] = 0.33333
        rgb = matplotlib.colors.hsv_to_rgb(hsl)
        if image_item is None:
            image_item = pg.ImageItem(rgb)
            view_box.addItem(image_item)
            graphics_view.resize(2*rgb.shape[0], 2*rgb.shape[1])
            graphics_view.show()
        image_item.updateImage(rgb)

SHOW_PLOT = True
# SHOW_PLOT = False
if not os.getenv('DISPLAY'):
    # But not if there is no x server:
    SHOW_PLOT = False

if SHOW_PLOT:
    import pyqtgraph as pg
    qapplication = QtGui.QApplication([])
    graphics_view = pg.GraphicsView()
    graphics_view.setWindowTitle('MPI task %d'%simulator.MPI_rank)
    view_box = pg.ViewBox()
    graphics_view.setCentralItem(view_box)
    view_box.setAspectLocked(True)
    image_item = None

def initial_guess(x, y):
    sigma_x = 0.5*R
    sigma_y = 0.5*R
    f = np.sqrt(np.exp(-x**2/(2*sigma_x**2) - y**2/(2*sigma_y**2)))
    return f

def run_sims():
    # import lineprofiler
    # lineprofiler.setup(outfile='lineprofile-%d.txt'%MPI_rank)

    psi = simulator.elements.make_vector(initial_guess)
    simulator.normalise(psi, N_2D)
    psi = simulator.successive_overrelaxation(psi, H, K_diags, mu, output_callback=plot, output_interval=100)

    # # Scatter some vortices randomly about.
    # # Ensure all MPI tasks agree on the location of the vortices, by
    # # seeding the pseudorandom number generator with the same seed in
    # # each process:
    # np.random.seed(42)
    # for i in range(30):
    #     sign = np.sign(np.random.normal())
    #     x_vortex = np.random.normal(0, scale=R)
    #     y_vortex = np.random.normal(0, scale=R)
    #     psi[:] *= np.exp(sign * 1j*np.arctan2(simulator.y - y_vortex, simulator.x - x_vortex))
    # psi = simulator.evolve(psi, V, t_final=400e-6, output_group='vortices', imaginary_time=True, output_callback=plot)

    # psi = simulator.evolve(psi, V, t_initial=t_initial, t_final=np.inf, output_group='time evolution',
    #                        output_callback=plot, output_interval=100, rk4=True, timestep_factor=1/pi)

if not SHOW_PLOT:
    run_sims()
else:
    # If we're plotting stuff, Qt will need the main thread:
    inthread(run_sims)

    import signal
    # Let the interpreter run every 500ms so it sees Ctrl-C interrupts:
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.
    # Upon seeing a ctrl-c interrupt, quit the event loop
    signal.signal(signal.SIGINT, lambda *args: qapplication.exit())

    qapplication.exec_()
