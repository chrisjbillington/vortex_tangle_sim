from pylab import plot, show, grid, subplot, ylim, title
import numpy as np


def get_legendre_polynomials(N):
    """Returns a list of numpy.poly1d objects representing the first N
    Legendre polynomials, normalised to unit L2 norm over [-1, 1]"""
    from scipy.special import legendre
    L = []
    for n in range(N):
        L_n = legendre(n)
        # Normalise:
        L_n /= L_n.normcoef
        L.append(L_n)
    return L


def get_quadrature_points_and_weights(N):
    """Returns the spatial points and weights for the N-point Gauss-Legendre
    quadrature. These are the spatial points at which all but one of the DVR
    basis functions is zero."""
    from scipy.special import p_roots
    points, weights = p_roots(N)
    return points, weights


def make_DVR_basis(N):
    """Returns a list of numpy.poly1d objects representing the basis functions
    of the discrete variable method using the N-point Gauss-Legendre
    quadrature. List is ordered to correspond to the points returned by
    get_quadrature_points_and_weights().

    It is easy to compute the basis functions from the Legendre polynomials,
    because the integral for their projection onto each Legendre polynomial is
    exactly given by the quadrature rule."""
    L = get_legendre_polynomials(N)
    x, w = get_quadrature_points_and_weights(N)
    u = []
    for i in range(N):
        u_i = np.poly1d([])
        for n in range(N):
            u_i += np.sqrt(w[i])*L[n](x[i])*L[n]
        u.append(u_i)
    return u


def make_DVR_differetial_operator(N, order=1):
    """"Return an NxN array of the matrix representation of the derivative
    operator of a given order in the N-point Gauss-Legendre DVR basis.

    The matrix elements for the operator are easy to compute, because the
    integrals for them are exactly given by the quadrature rule."""
    u = make_DVR_basis(N)
    x, w = get_quadrature_points_and_weights(N)
    # Differentiate the basis functions to the given order:
    dn_u_dxn = [u_i.deriv(order) for u_i in u]
    dn_dxn = np.zeros((N,N))
    for i, u_i in enumerate(u):
        for j, dn_u_j_dxn in enumerate(dn_u_dxn):
            # Compute one matrix element <u_i | dn_dxn u_j> using the quadrature rule:
            for x_n, w_n in zip(x, w):
                dn_dxn[i, j] += w_n*u_i(x_n)*dn_u_j_dxn(x_n)
    return dn_dxn


def make_DVR_vector(x_dense, psi_dense, N):
    """takes points x_dense and the values psi_dense of a function at those points, and
    returns an array of length N containing the coefficients for that
    function's representation in N-point Gauss-Legendre DVR basis. x should be
    dense in the domain [-1, 1]. linspace(-1,1,1000) should do it.

    Actually projects the function onto the basis vectors using full integrals
    rather than assuming the quadrature rule is valid and only using the values of
    the function at the quadrature points. This ensures we represent it as accurately as
    possible """
    psi = np.zeros(N)
    u = make_DVR_basis(N)
    for i, u_i in enumerate(u):
        psi[i] = np.trapz(psi_dense*u_i(x_dense), x_dense)
    return psi


def interpolate_DVR_vector(psi, x_dense):
    """Takes a vector psi in the Gauss-Legendre DVR basis and interpolates the spatial
    function it represents to the points in the array x_dense, which should be in
    the domain [-1, 1]"""
    f = np.zeros(len(x_dense))
    N = len(psi)
    u = make_DVR_basis(N)
    for psi_i, u_i in zip(psi, u):
        f += psi_i*u_i(x_dense)
    return f


def gradientn(y, dx, n=1):
    result = y
    for i in range(n):
        result = np.gradient(result, dx)
    return result


N = 10
differentiation_order = 1

L = get_legendre_polynomials(N)
x, w = get_quadrature_points_and_weights(N)
u = make_DVR_basis(N)
dn_dxn = make_DVR_differetial_operator(N, order=differentiation_order)


x_dense = np.linspace(-1,1,1000)
dx = x_dense[1] - x_dense[0]

psi_dense = np.exp(-x_dense**2)
psi = make_DVR_vector(x_dense, psi_dense, N)
psi_interpolated = interpolate_DVR_vector(psi, x_dense)

d_psi_dense_dx = gradientn(psi_dense, dx, differentiation_order)
d_psi_dx = np.dot(dn_dxn, psi)
d_psi_dx_interpolated = interpolate_DVR_vector(d_psi_dx, x_dense)

# Plot the Legendre polynomials:
subplot(221)
title('Legendre Polynomials')
for L_n in L:
    plot(x_dense, L_n(x_dense), 'b-')
grid(True)
ylim(-2,2)

# Plot the DVR basis functions:
subplot(222)
title('DVR basis polynomials')
for x_i, u_i in zip(x, u):
    plot(x_dense, u_i(x_dense), 'g-')
    plot(x_i, u_i(x_i), 'ko')
plot(x, np.zeros(len(x)), 'ko')
grid(True)
ylim(-1,4)

# Plot the wavefunction and its best representation:
subplot(223)
title('Actual and DVR interpolated Gaussian')
plot(x_dense, psi_dense, 'b-')
plot(x_dense, psi_interpolated, 'r-')
plot(x, psi/np.sqrt(w), 'ko')
grid(True)
ylim(-0,1)

# Plot the derivative of the wavefunction and its best representation:
subplot(224)
title('derivative of Gaussian with DVR differential operator')
plot(x_dense, d_psi_dense_dx, 'b-')
plot(x_dense, d_psi_dx_interpolated, 'r-')
plot(x, d_psi_dx/np.sqrt(w), 'ko')
grid(True)
ylim(-1,1)

show()
