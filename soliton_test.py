from __future__ import division
import numpy as np
import pylab as pl

from integrator import euler, rk4, HDFOutput
from FEDVR import FiniteElements
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

# Dense grid for plotting:
x = np.linspace(x_min, x_max, 1024)

dx = x[1]-x[0]

# Fourier space:
kx = 2*pi*pl.fftfreq(len(x),d=dx)

# Laplace operator in Fourier space:
f_laplacian = -(kx**2)


# Finite elements:
N = 4
n_elements = 50
elements = FiniteElements(N, n_elements, x_min, x_max)

# Second derivative operators:
grad2 = elements.second_derivative_operators()


def V(x):
    """The harmonic trap"""
    return 0.5*m*omega**2*x**2


def get_second_derivatives(psi):
    # Get the second derivatives of the vectors:
    grad2psi = []
    for psi_n, grad2_n in zip(psi, grad2):
        grad2psi_n = np.dot(grad2_n, psi_n)
        grad2psi.append(grad2psi_n)
    # Exchange at the edges:
    for left_vector, right_vector in zip(grad2psi, grad2psi[1:]):
        left_vector[-1] = right_vector[0] = left_vector[-1] + right_vector[0]
    return grad2psi


def get_nonlinear_terms(psi):
    modsquaredpsi = []
    for element, psi_n in zip(elements.elements, psi):
        modsquaredpsi_n = np.zeros(N)
        for i, (point, basis_function) in enumerate(zip(element.points, element.basis)):
            value = psi_n[i]*basis_function(point)
            modsquaredpsi_n[i] = np.abs(value)**2
        modsquaredpsi.append(modsquaredpsi_n)
    return modsquaredpsi


def plot(i, t, psi_normal, *psi):
    # pl.plot(x*1e6, np.abs(psi_normal)**2, 'k-')
    # if psi:
    #     psi_interp, points, values = elements.interpolate_vectors(psi, x)
    #     pl.plot(x*1e6, np.abs(psi_interp)**2, 'r--')
    #     pl.plot(points*1e6, np.abs(values)**2, 'ko')

    grad2psi = get_second_derivatives(psi)
    grad2psi_interp, points, grad2values = elements.interpolate_vectors(grad2psi, x)
    modsquaredpsi = get_nonlinear_terms(psi)
    psi_interp, points, values = elements.interpolate_vectors(psi, x)
    mod2psi_interp = np.abs(psi_interp)**2
    mod2values = np.abs(values)**2
    f_psi_normal = pl.fft(psi_normal)
    f_grad2psi = f_laplacian*f_psi_normal
    grad2psi_normal = pl.ifft(f_grad2psi).real
    modsquared_psi_normal = np.abs(psi_normal)**2


    # derivs = dpsi_dt(t, psi_normal, *psi)
    # time_deriv_normal = derivs[0]
    # time_deriv = derivs[1:]
    # time_deriv_interp, points, values = elements.interpolate_vectors(time_deriv, x)
    # pl.plot(x*1e6, time_deriv_interp, 'r--')
    # pl.plot(points*1e6, values, 'ko')
    # pl.plot(x*1e6, time_deriv_normal, 'k-')

    pl.plot(x*1e6, modsquared_psi_normal, 'k-')
    pl.plot(x*1e6, mod2psi_interp, 'r--')
    pl.plot(points*1e6, mod2values, 'ko')

    # pl.plot(x*1e6, grad2psi_normal, 'k-')
    # pl.plot(x*1e6, grad2psi_interp, 'r--')
    # pl.plot(points*1e6, grad2values, 'ko')

    # for mod2psi_n, element in zip(modsquaredpsi, elements.elements):
    #     pl.plot(element.points*1e6, mod2psi_n, 'ko')

    for boundary in elements.boundaries:
        pl.axvline(boundary*1e6, linestyle='--', color='k')
    pl.grid(True)
    pl.axis([-1.0, 1, 0, 2.6e20])
    pl.savefig('output.png')
    # pl.show()
    pl.clf()

def renormalise(i, t, psi_normal, *psi):
    # imposing normalisation on the total wavefunction:
    if psi:
        ncalc = 0
        for psi_n in psi:
            ncalc += np.dot(psi_n[:-1].conj().T, psi_n[:-1])
        for psi_n, element in zip(psi, elements.elements):
            points = element.points
            psi_n[:] *= np.sqrt(N_1D/ncalc.real)
            psi_n[points >= 0] = np.abs(psi_n[points >= 0])
            psi_n[points < 0] = -np.abs(psi_n[points < 0])
            psi_n[points == 0] = 0
    ncalc_normal = (np.abs(psi_normal)**2).sum()*dx
    psi_normal[:] = np.abs(psi_normal)*np.sqrt(N_1D/ncalc_normal.real)
    psi_normal[x >= 0] = np.abs(psi_normal[x >= 0])
    psi_normal[x < 0] = -np.abs(psi_normal[x < 0])


def initial():
    """Imaginary time evolution for the initial state"""

    dt = 2e-7
    t_final = 30e-3

    # The initial guess:
    def initial_guess(x):
        # Thomas-Fermi distribution with a phase jump in the middle
        # psi = rho_max*(1 - x**2/R**2)
        # psi = np.clip(psi, 0, None)
        # psi = np.sqrt(psi)
        # # psi *= (x<0) - 1*(x>=0)
        return np.exp(-x**2/(0.25*R**2))
        # return psi

    psi = elements.make_vectors(initial_guess)
    psi_normal = initial_guess(x)

    global dpsi_dt
    def dpsi_dt(t, psi_normal, *psi):
        """The differential equations for imaginary time evolution"""

        grad2psi = get_second_derivatives(psi)
        modsquaredpsi = get_nonlinear_terms(psi)

        d_psi_dt = []
        for psi_n, modsquaredpsi_n, grad2psi_n, element in zip(psi, modsquaredpsi, grad2psi, elements.elements):
            points = element.points
            d_psi_n_dt = hbar/(2*m)*grad2psi_n - 1/hbar*(V(points) + g*modsquaredpsi_n - mu)*psi_n
            # d_psi_n_dt = hbar/(2*m)*grad2psi_n - 1/hbar*(V(points))*psi_n
            d_psi_dt.append(d_psi_n_dt)

        # Hop over into Fourier space:
        f_psi_normal = pl.fft(psi_normal)

        # Calculate grad squared of psi there:
        f_grad2psi = f_laplacian*f_psi_normal

        # Hop back into real space:
        grad2psi_normal = pl.ifft(f_grad2psi).real

        # Calculate dpsi_dt in real space:
        d_psi_normal_dt = hbar/(2*m)*grad2psi_normal - 1/hbar*(V(x) + g*np.abs(psi_normal)**2 - mu)*psi_normal
        # d_psi_normal_dt = hbar/(2*m)*grad2psi_normal - 1/hbar*(V(x))*psi_normal
        # d_psi_normal_dt = np.zeros(len(x))
        return [d_psi_normal_dt] + d_psi_dt

    # Creating a dictionary of triggers that will be called repeatedly
    # during integration. The function output.sample will be called
    # every 50 steps:
    routines = {1: renormalise, 2000:plot}

    # Start the integration:
    euler(dt, t_final, dpsi_dt, [psi_normal] + psi, routines = routines)

if __name__ == '__main__':
    initial()
