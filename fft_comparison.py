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

x = np.linspace(x_min, x_max, 256, endpoint=False)
y = np.linspace(y_min, y_max, 256, endpoint=False)
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
    global image_item
    rho_plot = np.abs(psi)**2
    if image_item is None:
        image_item = image_view.setImage(rho_plot)
    else:
        image_item.setData(rho_plot)


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

    def check_convergence(i,t,psi):
        mucalc, u_mucalc = compute_mu(psi)
        convergence = abs(u_mucalc/mucalc)
        print(i, t, mucalc, convergence, 1e3*(time.time() - start_time)/(i+1), 'msps')

    # Creating a dictionary of triggers that will be called repeatedly during integration. The function
    # renormalise will be called every step, and the function output.sample will be called every 50
    # steps:
    routines = {1: renormalise, 10: plot, 100:check_convergence}

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

    import pyqtgraph as pg

    qapplication = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.resize(800,800)
    image_view = pg.ImageView()
    image_item = None
    win.setCentralWidget(image_view)
    win.show()
    win.setWindowTitle('pyqtgraph example: ImageView')

    def run_sims():
        import lineprofiler
        lineprofiler.setup()
        import cPickle as pickle

        if not os.path.exists('initial_fft.pickle'):
            # Initial guess:
            sigmaguess = 0.5*R
            psi = np.sqrt(N_2D/(2 * pi * sigmaguess**2)) * np.exp(-((X)**2 + (Y)**2)/(4 * sigmaguess**2))
            renormalise(None, None, psi)
            psi = initial(psi, dt=5e-7, t_final=30e-3)
            with open('initial_fft.pickle', 'w') as f:
                psi = pickle.dump(psi, f)
        else:
            with open('initial_fft.pickle') as f:
                psi = pickle.load(f)

        psi = np.array(psi, dtype=complex)

        if not os.path.exists('vortices_fft.pickle'):
            # Scatter some vortices randomly about:
            for i in range(30):
                x_vortex = np.random.normal(0, scale=R)
                y_vortex = np.random.normal(0, scale=R)
                psi[:] *= np.exp(1j*np.arctan2(Y - y_vortex, X - x_vortex))
            print('doing a thing')
            psi = initial(psi, dt=1e-7, t_final=400e-6)
            with open('vortices_fft.pickle', 'w') as f:
                psi = pickle.dump(psi, f)
        else:
            with open('vortices_fft.pickle') as f:
                psi = pickle.load(f)
        psi = evolution(psi, dt=2e-7, t_final=np.inf)


    # run_sims()
    inthread(run_sims)
    qapplication.exec_()
