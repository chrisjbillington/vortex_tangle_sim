from __future__ import division
from pylab import plot, show, grid, subplot, ylim, title
import numpy as np


def gauss_legendre_points_and_weights(N):
    """Returns the spatial points and weights for the N-point Gauss-Legendre
    quadrature."""
    from scipy.special import p_roots
    points, weights = p_roots(N)
    return points, weights


def gauss_lobatto_points_and_weights(N, x_l=-1, x_r=1):
    """Returns the spatial points and weights for the N-point Gauss-Lobatto
    quadrature."""
    from scipy.special import legendre
    points = np.zeros(N)
    weights = np.zeros(N)
    # Endpoints first:
    points[0] = -1
    points[-1] = 1
    weights[0] = weights[-1] = 2/(N*(N - 1))
    # Interior points are given by the roots of P'_{n-1}:
    P = legendre(N-1)
    points[1:-1] = sorted(P.deriv().r)
    # And their weights:
    weights[1:-1] = 2/(N*(N - 1)*P(points[1:-1])**2)
    return points, weights


def make_DVR_basis_polynomials(x, w):
    """Returns a list of numpy Polynomial objects representing the basis
    functions of the discrete variable representation using the the Gauss
    quadrature with points x and weights w."""
    from numpy.polynomial.polynomial import Polynomial, polyfromroots
    u = []
    for x_i, w_i in zip(x, w):
        # Each DVR basis function is simply a polynomial with zeros at all the
        # other points, scaled to equal 1/sqrt(w_i) at its own point.
        other_points = np.array([x_q for x_q in x if x_q != x_i])
        u_i = 1/np.sqrt(w_i)*Polynomial(polyfromroots(other_points))/np.prod(x_i - other_points)
        u.append(u_i)
    return u


class DVRBasis(object):
    def __init__(self, N, x_l=-1, x_r=1, quadrature='lobatto'):
        """A class for operations with the N-point discrete variable representation
        basis for either Gauss-Legendre or Gauss-Lobatto quadrature on an
        interval [x_l, x_r].

        Attributes:
            N:     number of quadrature points/basis functions
            x_l:   left boundary of interval
            x_r:   right boundary of interval
            x:     quadrature points
            w      quadrature weights
            u      DVR basis polynomials, valid in in the interval [x_l, x_r]

        """
        self.N = N
        self.x_l = x_l
        self.x_r = x_r
        self.quadrature = quadrature
        if self.quadrature == 'legendre':
            self.x, self.w = gauss_lobatto_points_and_weights(N)
        elif self.quadrature == 'lobatto':
            self.x, self.w = gauss_lobatto_points_and_weights(N)
        else:
            raise ValueError('quadrature must be either \'lobatto\' or \'legendre\'')
        # Shift the quadrature points from the interval [-1, 1] to [x_l, x_r], and
        # scale the weights appropriately:
        self.x = (self.x + 1)*(x_r - x_l)/2 + x_l
        self.w /= (x_r - x_l)/2
        # Make the DVR basis functions:
        self.u = make_DVR_basis_polynomials(self.x, self.w)

    def make_vector(self, x_dense, psi_dense):
        """Takes points x_dense and the values psi_dense of a function at those
        points, and returns an array containing the coefficients for that
        function's representation in that DVR basis. Only the part of the function
        in the interval [self.x_l, self.x_r] is used."""
        psi = np.zeros(self.N)
        valid = (self.x_l <= x_dense) & (x_dense <= self.x_r)
        x_dense = x_dense[valid]
        psi_dense = psi_dense[valid]
        for i, u_i in enumerate(self.u):
            psi[i] = np.trapz(psi_dense*u_i(x_dense), x_dense)
        return psi

    def interpolate_vector(self, psi, x_dense):
        """Takes a vector psi in the DVR basis and interpolates the spatial
        function it represents to the points in the array x_dense"""
        f = np.zeros(len(x_dense))
        for psi_i, u_i in zip(psi, self.u):
            try:
                f += psi_i*u_i(x_dense)
            except:
                import IPython
                IPython.embed()
        # Clip outside the valid region:
        valid = (self.x_l <= x_dense) & (x_dense <= self.x_r)
        f[~valid] = 0
        return f

    def differential_operator(self, order=1):
        """"Return an len(x) x len(x) array for the matrix representation of the derivative
        operator of a given order in the DVR basis for points x and weights w.

        The matrix elements for the operator are easy to compute, because the
        integrals for them are exactly given by the quadrature rule."""
        # Differentiate the basis functions to the given order:
        dn_u_dxn = [u_i.deriv(order) for u_i in self.u]
        dn_dxn = np.zeros((self.N,self.N))
        for i, u_i in enumerate(self.u):
            for j, dn_u_j_dxn in enumerate(dn_u_dxn):
                # Compute one matrix element <u_i | dn_dxn u_j> using the quadrature rule:
                for x_n, w_n in zip(self.x, self.w):
                    dn_dxn[i, j] += w_n*u_i(x_n)*dn_u_j_dxn(x_n)
        return dn_dxn


def gradientn(y, dx, n=1):
    result = y
    for i in range(n):
        result = np.gradient(result, dx)
    return result


def test_single_element():
    N = 7
    differentiation_order = 1

    # Get our quadrature points and weights, our DVR basis functions, and
    # our differential operator in the DVR basis:
    dvr_basis = DVRBasis(N, -1, 1, 'lobatto')
    x = dvr_basis.x
    w = dvr_basis.w
    u = dvr_basis.u
    dn_dxn = dvr_basis.differential_operator(differentiation_order)

    # A dense grid for making plots
    x_dense = np.linspace(-1,1,1000)
    dx = x_dense[1] - x_dense[0]

    # Our Gaussian wavefunction, its representation in the DVR basis,
    # and that representation's interpolation back onto the dense grid:
    psi_dense = np.exp(-x_dense**2)
    psi = dvr_basis.make_vector(x_dense, psi_dense)
    psi_interpolated = dvr_basis.interpolate_vector(psi, x_dense)

    # The derivative of our Gaussian wavefunction, the derivative of its
    # representation in the DVR basis (computed with the DVR derivative
    # operator), and that derivatives interpolation back onto the dense grid:
    d_psi_dense_dx = gradientn(psi_dense, dx, differentiation_order)
    d_psi_dx = np.dot(dn_dxn, psi)
    d_psi_dx_interpolated = dvr_basis.interpolate_vector(d_psi_dx, x_dense)

    # Plot the DVR basis functions:
    subplot(211)
    title('DVR basis polynomials')
    for x_i, u_i in zip(x, u):
        plot(x_dense, u_i(x_dense))
        plot(x_i, u_i(x_i), 'ko')
    plot(x, np.zeros(len(x)), 'ko')
    grid(True)
    ylim(-1,6)

    # Plot the wavefunction and its interpolated DVR representation:
    subplot(223)
    title('Exact and DVR Gaussian')
    plot(x_dense, psi_dense, 'b-')
    plot(x_dense, psi_interpolated, 'r-')
    plot(x, psi/np.sqrt(w), 'ko')
    grid(True)
    ylim(-0,1)

    # Plot the derivative of the wavefunction and its interpolated DVR representation:
    subplot(224)
    title('Exact and DVR derivative')
    plot(x_dense, d_psi_dense_dx, 'b-')
    plot(x_dense, d_psi_dx_interpolated, 'r-')
    plot(x, d_psi_dx/np.sqrt(w), 'ko')
    grid(True)
    ylim(-1,1)

    show()


test_single_element()
