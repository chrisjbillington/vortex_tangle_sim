from __future__ import division
from pylab import plot, show, grid, subplot, ylim, title
import numpy as np


def gauss_legendre_points_and_weights(N):
    """Returns the spatial points and weights for the N-point Gauss-Legendre
    quadrature. These are the spatial points at which all but one of the
    corresponding DVR basis functions are zero."""
    from scipy.special import p_roots
    points, weights = p_roots(N)
    return points, weights


def gauss_lobatto_points_and_weights(N):
    """Returns the spatial points and weights for the N-point Gauss-Lobatto
    quadrature. These are the spatial points at which all but one of the
    corresponding DVR basis functions are zero."""
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


def make_DVR_basis(x, w):
    """Returns a list of numpy Polynomial objects representing the basis functions
    of the discrete variable representation using the the Gauss quadrature with points
    x and weights w."""
    from numpy.polynomial.polynomial import Polynomial, polyfromroots
    u = []
    for x_i, w_i in zip(x, w):
        # Each DVR basis function is simply a polynomial with zeros at all the
        # other points, scaled to equal 1/sqrt(w_i) at its own point.
        other_points = np.array([x_q for x_q in x if x_q != x_i])
        u_i = 1/np.sqrt(w_i)*Polynomial(polyfromroots(other_points))/np.prod(x_i - other_points)
        u.append(u_i)
    return u


def make_DVR_differetial_operator(x, w, order=1):
    """"Return an len(x) x len(x) array for the matrix representation of the derivative
    operator of a given order in the DVR basis for points x and weights w.

    The matrix elements for the operator are easy to compute, because the
    integrals for them are exactly given by the quadrature rule."""
    N = len(x)
    u = make_DVR_basis(x, w)
    # Differentiate the basis functions to the given order:
    dn_u_dxn = [u_i.deriv(order) for u_i in u]
    dn_dxn = np.zeros((N,N))
    for i, u_i in enumerate(u):
        for j, dn_u_j_dxn in enumerate(dn_u_dxn):
            # Compute one matrix element <u_i | dn_dxn u_j> using the quadrature rule:
            for x_n, w_n in zip(x, w):
                dn_dxn[i, j] += w_n*u_i(x_n)*dn_u_j_dxn(x_n)
    return dn_dxn


def make_DVR_vector(x_dense, psi_dense, x, w):
    """Takes points x_dense and the values psi_dense of a function at those
    points, and the quadrature points x and weights w for a DVR basis, and
    returns an array of length len(x) containing the coefficients for that
    function's representation in that DVR basis. x_dense should be dense in
    the domain [-1, 1]. linspace(-1,1,1000) should do it.

    Actually projects the function onto the basis vectors using full integrals
    rather than assuming the quadrature rule is valid and only using the
    values of the function at the quadrature points. This ensures we represent
    it as accurately as possible """
    psi = np.zeros(len(x))
    u = make_DVR_basis(x, w)
    for i, u_i in enumerate(u):
        psi[i] = np.trapz(psi_dense*u_i(x_dense), x_dense)
    return psi


def interpolate_DVR_vector(psi, x, w, x_dense):
    """Takes a vector psi in the DVR basis with quadrature points x and
    weights w, and interpolates the spatial function it represents to the
    points in the array x_dense, which should be in the domain [-1, 1]"""
    f = np.zeros(len(x_dense))
    u = make_DVR_basis(x, w)
    for psi_i, u_i in zip(psi, u):
        f += psi_i*u_i(x_dense)
    return f


def gradientn(y, dx, n=1):
    result = y
    for i in range(n):
        result = np.gradient(result, dx)
    return result


def main():
    N = 7
    differentiation_order = 1

    # Get our quadrature points and weights, our DVR basis functions, and
    # our differential operator in the DVR basis:
    x, w = gauss_lobatto_points_and_weights(N)
    u = make_DVR_basis(x, w)
    dn_dxn = make_DVR_differetial_operator(x, w, order=differentiation_order)

    # A dense grid for making plots
    x_dense = np.linspace(-1,1,1000)
    dx = x_dense[1] - x_dense[0]

    # Our Gaussian wavefunction, its representation in the DVR basis,
    # and that representation's interpolation back onto the dense grid:
    psi_dense = np.exp(-x_dense**2)
    psi = make_DVR_vector(x_dense, psi_dense, x, w)
    psi_interpolated = interpolate_DVR_vector(psi, x, w, x_dense)

    # The derivative of our Gaussian wavefunction, the derivative of its
    # representation in the DVR basis (computed with the DVR derivative
    # operator), and that derivatives interpolation back onto the dense grid:
    d_psi_dense_dx = gradientn(psi_dense, dx, differentiation_order)
    d_psi_dx = np.dot(dn_dxn, psi)
    d_psi_dx_interpolated = interpolate_DVR_vector(d_psi_dx, x, w, x_dense)

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

main()
