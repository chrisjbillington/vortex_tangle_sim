from __future__ import division, print_function
import pylab as pl
import numpy as np

from autofftw import fftw, ifftw
from integrator import euler, rk4

from PyQt4 import QtGui
from qtutils import inthread, inmain_decorator

import os, time

def get_number_and_trap(rhomax, R):
    N = pi*rhomax*R**2/2
    omega = np.sqrt(2*g*rhomax/(m*R**2))
    return N, omega

# Constants:
pi = np.pi
hbar = 1.054572e-34                         # Reduced Planck's constant
a_0  = 5.29177209e-11                       # Bohr radius
u    = 1.660539e-27                         # unified atomic mass unit
m  = 86.909180*u                            # 87Rb atomic mass
a  = 98.98*a_0                              # 87Rb |2,2> scattering length
g  = 4*pi*hbar**2*a/m                       # 87Rb self interaction constant
rhomax = 2.5e14*1e6
R = 7.5e-6
N_2D, omega = get_number_and_trap(rhomax, R)


# Space:
x_min = -10e-6
x_max = 10e-6
y_min = -10e-6
y_max = 10e-6

nx = 512
ny = 512

x = np.linspace(x_min, x_max, nx, endpoint=False)
y = np.linspace(y_min, y_max, ny, endpoint=False)
X,Y = np.meshgrid(x,y)

dx = x[1]-x[0]
dy = y[1]-y[0]

# Fourier space:
kx = 2*pi*pl.fftfreq(len(x),d=dx)
ky = 2*pi*pl.fftfreq(len(y),d=dy)
Kx,Ky = np.meshgrid(kx,ky)

# Laplace operator in Fourier space:
f_laplacian = -(Kx**2 + Ky**2)

# The Harmonic trap at our gridpoints, (n_elements x N):
V = 0.5*m*omega**2*(X**2 + Y**2)

@inmain_decorator()
def plot(i, t, psi):
    if SHOW_PLOT:
        rho_plot = np.abs(psi)**2
        phase_plot = np.angle(psi)
        density_image_view.setImage(rho_plot)
        phase_image_view.setImage(phase_plot)


def compute_mu(psi):
    density = np.abs(psi)**2
    f_psi = fftw(psi)
    f_grad2psi = f_laplacian*f_psi
    grad2psi = ifftw(f_grad2psi).real
    mu = -hbar**2/(2*m)*grad2psi/psi + V + g*density
    av_mu = np.average(mu[density > 0.000001*density.max()]).real
    u_mu = np.std(mu[density > 0.000001*density.max()])
    return av_mu, u_mu


def compute_number(psi):
    ncalc = np.vdot(psi, psi).real*dx*dy
    return ncalc


def renormalise(i,t,psi):
    # imposing normalisation on both components:
    ncalc = (np.abs(psi)**2).sum()*dx*dy
    psi[:] *= np.sqrt(N_2D/ncalc.real)


def initial(psi, dt, t_final):

    def dpsi_dt(t,psi):
        """The differential equations for imaginary time evolution"""
        # Hop over into Fourier space:
        f_psi = fftw(psi)

        # Calculate grad squared of psi there:
        f_grad2psi = f_laplacian*f_psi

        # Hop back into real space:
        grad2psi = ifftw(f_grad2psi)

        # Calculate dpsi_dt in real space:
        d_psi_dt = hbar/(2*m)*grad2psi - 1/hbar*(V + g*np.abs(psi)**2)*psi

        return d_psi_dt

    def check_convergence_and_plot(i,t,psi):
        mucalc, u_mucalc = compute_mu(psi)
        convergence = abs(u_mucalc/mucalc)
        print(i, t, mucalc, convergence, 1e3*(time.time() - start_time)/(i+1), 'msps')
        plot(i, t, psi)

    # Creating a dictionary of triggers that will be called repeatedly during integration. The function
    # renormalise will be called every step, and the function output.sample will be called every 50
    # steps:
    routines = {1: renormalise, 100:check_convergence_and_plot}

    # Start the integration:
    start_time = time.time()
    euler(dt, t_final, dpsi_dt, [psi], routines = routines)
    return psi

def evolution(psi, dt, t_final):

    def dpsi_dt(t, psi):
        """The differential equations for time evolution of the condensate"""
        # Hop over into Fourier space:
        f_psi = fftw(psi)

        # Calculate grad squared of psi there:
        f_grad2psi = f_laplacian*f_psi

        # Hop back into real space:
        grad2psi = ifftw(f_grad2psi)

        # Calculate dpsi_dt in real space:
        d_psi_dt = 1j*hbar/(2*m)*grad2psi -  1j/hbar*(V + g*np.abs(psi)**2)*psi

        return d_psi_dt

    def print_stuff_and_plot(i,t,psi):
        mucalc, u_mucalc = compute_mu(psi)
        ncalc = compute_number(psi)
        step_rate = (i+1)/(time.time() - start_time)
        print(round(t*1e6), 'us',
              round(np.log10(abs(ncalc/n_initial-1))),
              round(np.log10(abs(mucalc/mu_initial-1))),
              round(1e3/step_rate, 1), 'msps')
        plot(i, t, psi)

    mu_initial, u_mu_initial = compute_mu(psi)
    n_initial = compute_number(psi)
    # Creating a dictionary of triggers that will be called repeatedly
    # during integration. The function output.sample will be called
    # every 50 steps:
    routines = {100:print_stuff_and_plot}

    # Start the integration:
    start_time = time.time()
    rk4(dt,t_final,dpsi_dt, [psi], routines = routines)


if __name__ == '__main__':

    SHOW_PLOT = False
    if SHOW_PLOT:
        import pyqtgraph as pg

        qapplication = QtGui.QApplication([])
        win = QtGui.QWidget()
        win.resize(800,800)
        density_image_view = pg.ImageView()
        phase_image_view = pg.ImageView()
        layout = QtGui.QVBoxLayout(win)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(density_image_view)
        layout.addWidget(phase_image_view)
        win.show()
        win.setWindowTitle('FFT method')

    def run_sims():
        import cPickle as pickle
        specs = (nx, ny)
        initial_filename = 'cache/FFT_initial_(%dx%d).pickle'%specs
        vortices_filename = 'cache/FFT_vortices_(%dx%d).pickle'%specs

        if not os.path.exists(initial_filename):
            # Initial guess:
            sigmaguess = 0.5*R
            psi = np.sqrt(N_2D/(2 * pi * sigmaguess**2)) * np.exp(-((X)**2 + (Y)**2)/(4 * sigmaguess**2))
            psi = np.array(psi, dtype=complex)
            renormalise(None, None, psi)
            while True:
                psi = initial(psi, dt=2e-7, t_final=1e-3)
                with open(initial_filename, 'w') as f:
                    pickle.dump(psi, f)
        else:
            with open(initial_filename) as f:
                psi = pickle.load(f)

        if not os.path.exists(vortices_filename):
            np.random.seed(42)
            # Scatter some vortices randomly about.
            for i in range(30):
                sign = np.sign(np.random.normal())
                x_vortex = np.random.normal(0, scale=R)
                y_vortex = np.random.normal(0, scale=R)
                psi[:] *= np.exp(sign * 1j*np.arctan2(Y - y_vortex, X - x_vortex))
            psi = initial(psi, dt=5.86e-8, t_final=400e-6)
            with open(vortices_filename, 'w') as f:
                pickle.dump(psi, f)
        else:
            with open(vortices_filename) as f:
                psi = pickle.load(f)
        psi = evolution(psi, dt=5.86e-8, t_final=np.inf)


    if SHOW_PLOT:
        inthread(run_sims)
        qapplication.exec_()
    else:
        run_sims()
