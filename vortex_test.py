from __future__ import division, print_function
import os
import cPickle as pickle

import numpy as np
from PyQt4 import QtCore, QtGui
from qtutils import inthread, inmain_decorator

import matplotlib

from BEC2D import BEC2DSimulator

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
n_elements_x_global = 32
n_elements_y_global = 32

# Number of DVR basis functions per element:
Nx = 7
Ny = 7

simulator = BEC2DSimulator(m, g, x_min_global, x_max_global, y_min_global, y_max_global, Nx, Ny,
                           n_elements_x_global, n_elements_y_global, output_file = 'vortex_test.h5')
x = simulator.x
y = simulator.y

# The Harmonic trap at our gridpoints, (n_elements x N):
V = 0.5*m*omega**2*(x**2 + y**2)

@inmain_decorator()
def plot(psi, output_log):
    if SHOW_PLOT:
        global image_item
        x_plot, y_plot, psi_interp = simulator.elements.interpolate_vector(psi, Nx, Ny)
        rho = np.abs(psi_interp)**2
        phase = np.angle(psi_interp)

        hsl = np.zeros(psi_interp.shape + (3,))
        hsl[:, :, 2] = rho/rho.max()
        hsl[:, :, 0] = np.array((phase + pi)/(2*pi))
        hsl[:, :, 1] = 0.33333
        rgb = matplotlib.colors.hsv_to_rgb(hsl)
        if image_item is None:
            image_item = pg.ImageItem(rgb)
            image_view.addItem(image_item)
            image_window.resize(*rgb.shape[:-1])
            image_window.show()
        image_item.updateImage(rgb)

SHOW_PLOT = True
# SHOW_PLOT = False
if not os.getenv('DISPLAY'):
    # But not if there is no x server:
    SHOW_PLOT = False

if SHOW_PLOT:
    import pyqtgraph as pg
    qapplication = QtGui.QApplication([])
    image_window = pg.GraphicsView()
    image_window.setWindowTitle('MPI task %d'%simulator.MPI_rank)
    image_view = pg.ViewBox()
    image_window.setCentralItem(image_view)
    image_view.setAspectLocked(True)
    image_item = None

def initial_guess(x, y):
        sigma_x = 0.5*R
        sigma_y = 0.5*R
        f = np.sqrt(np.exp(-x**2/(2*sigma_x**2) - y**2/(2*sigma_y**2)))
        return f

def run_sims():
    # import lineprofiler
    # lineprofiler.setup(outfile='lineprofile-%d.txt'%MPI_rank)

    if os.path.exists('psi.pickle'):
        with open('psi.pickle') as f:
            psi = pickle.load(f)
    else:
        psi = simulator.elements.make_vector(initial_guess)
        simulator.normalise(psi, N_2D)
        psi = simulator.find_groundstate(psi, V, mu, output_group='initial', output_interval=10, output_callback=plot)

        # Scatter some vortices randomly about.
        # Ensure all MPI tasks agree on the location of the vortices, by
        # seeding the pseudorandom number generator with the same seed in
        # each process:
        np.random.seed(42)
        for i in range(30):
            sign = np.sign(np.random.normal())
            x_vortex = np.random.normal(0, scale=R)
            y_vortex = np.random.normal(0, scale=R)
            psi[:] *= np.exp(sign * 1j*np.arctan2(simulator.y - y_vortex, simulator.x - x_vortex))
        psi = simulator.evolve(psi, V, t_final=400e-6, output_group='vortices', imaginary_time=True, output_callback=plot)
        with open('psi.pickle', 'w') as f:
            pickle.dump(psi, f)
    # Evolve in time:
    import time
    start_time = time.time()
    psi = simulator.evolve(psi, V, t_final=400e-6, output_group=None, output_callback=plot, output_interval=10)
    print('time taken:', time.time() - start_time)

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
