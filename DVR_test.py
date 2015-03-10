from __future__ import division, print_function
from pylab import figure, plot, show, grid, subplot, ylim, title, axvline, tight_layout
from numpy.polynomial.polynomial import Polynomial
import numpy as np

def gauss_lobatto_points_and_weights(N, left_boundary, right_boundary):
    """Returns the spatial points and weights for the N-point Gauss-Lobatto
    quadrature, scaled for the spatial interval [left_boundary, right_boundary]."""
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
    # Now we scale them from [-1, 1 ] to our domain:
    points = (points + 1)*(right_boundary - left_boundary)/2 + left_boundary
    weights *= (right_boundary - left_boundary)/2
    return points, weights


def lobatto_shape_functions(x):
    """Returns a list of Polynomial objects representing the Lobatto
    shape functions for the set of points x."""
    f = []
    for x_i in x:
        # The Lobatto shape function corresponding to point x_i is simply a Lagrange
        # polynomial with zeros at all the other points:
        other_points = np.array([x_q for x_q in x if x_q != x_i])
        f_i = Polynomial.fromroots(other_points)/np.prod(x_i - other_points)
        f.append(f_i)
    return f


class ElementFunction(object):
    """A callable representing one of the DVR functions within an element (not
    a bridge function). Evaluates to zero outside the element."""
    def __init__(self, shapefunction, weight, element_left_boundary, element_right_boundary):
        self.polynomial = shapefunction/np.sqrt(weight)
        self.weight = weight
        self.left_boundary = element_left_boundary
        self.right_boundary = element_right_boundary

    def __call__(self, arg):
        """Return the result of the polynomial in [self.element_left_boundary,
        self.element_right_boundary], and zero outside of it"""
        result = self.polynomial(arg)
        valid = (self.left_boundary <= arg) & (arg <= self.right_boundary)
        if isinstance(arg, np.ndarray):
            result[~valid] = 0
        elif not valid:
            result = 0
        return result


class BridgeFunction(object):
    """A callable representing a bridge function, one of the DVR functions
    joining two elements. Evaluates to zero outside the two elements it
    bridges."""
    def __init__(self, left_shapefunction, right_shapefunction,
                 left_weight, right_weight,
                 left_element_left_boundary, right_element_right_boundary,
                 x_bridge):
        self.left_segment = left_shapefunction/np.sqrt(left_weight + right_weight)
        self.right_segment = right_shapefunction/np.sqrt(left_weight + right_weight)
        self.left_boundary = left_element_left_boundary
        self.right_boundary = right_element_right_boundary
        self.x = x_bridge

    def __call__(self, arg):
        """Return the result of the polynomial in [x_l, x_r], and zero outside
        of it"""
        left_valid = (self.left_boundary <= arg) & (arg < self.x)
        right_valid = (arg >= self.x) & (arg <= self.right_boundary)
        if isinstance(arg, np.ndarray):
            result = np.zeros(len(arg))
            result[left_valid] = self.left_segment(arg[left_valid])
            result[right_valid] = self.right_segment(arg[right_valid])
        elif left_valid:
            result = self.left_segment(arg)
        elif right_valid:
            result = self.right_segment(arg)
        else:
            result = 0
        return result


class Element(object):
    def __init__(self, N, left_boundary=-1, right_boundary=1,
                 N_left=None, N_right=None, width_left=2, width_right=2):
        """A class for operations with the N-point discrete variable
        representation basis for Gauss-Lobatto quadrature on an interval
        [left_boundary, right_boundary]. If this DVR basis is just one element
        of many finite elements, then N_left and N_right specify how many DVR
        basis functions the elements to the left and right of this one have,
        and width_left and width_right specify the widths of those elements.
        This is important for making the bridge functions at the edges. On
        each side of an element boundary, both Elements have a copy of the
        bridge function, so care must be taken when combining elements
        together in some way not to double-count them.

        Attributes:
            N:              number of quadrature points/basis functions
            left_boundary:  left boundary of interval
            right_boundary: right boundary of interval
            points:         quadrature points
            weights:        quadrature weights
            basis:          DVR basis polynomials, valid in in the interval
                            [left_boundary, right_boundary], except for bridge functions
                            which are valid also over the adjacent element.

        """
        self.N = N
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.points, self.weights = gauss_lobatto_points_and_weights(N, left_boundary, right_boundary)
        # Make the DVR basis functions. Start with the Lobatto shape functions:
        shapefunctions = lobatto_shape_functions(self.points)
        self.basis = []

        # First consider the leftmost basis function:
        if N_left is not None:
            # Then the leftmost basis function is a bridge function:
            left_points, left_weights = gauss_lobatto_points_and_weights(
                                            N_left, left_boundary - width_left, left_boundary)
            left_shapefunctions = lobatto_shape_functions(left_points)
            basis_function = BridgeFunction(left_shapefunctions[-1], shapefunctions[0],
                                            left_weights[-1], self.weights[0],
                                            left_boundary - width_left, right_boundary,
                                            self.points[0])
        else:
            # It's a normal, non bridge basis function, an 'element function':
            basis_function = ElementFunction(shapefunctions[0], self.weights[0],
                                             left_boundary, right_boundary)
        self.basis.append(basis_function)

        # Now all the internal basis functions:
        for shapefunction, weight in zip(shapefunctions, self.weights)[1:-1]:
            basis_function = ElementFunction(shapefunction, weight, left_boundary, right_boundary)
            self.basis.append(basis_function)

        # Now the rightmost basis function:
        if N_right is not None:
            # Then the rightmost basis function is a bridge function:
            right_points, right_weights = gauss_lobatto_points_and_weights(
                                              N_right, right_boundary, right_boundary + width_right)
            right_shapefunctions = lobatto_shape_functions(right_points)
            basis_function = BridgeFunction(shapefunctions[-1], right_shapefunctions[0],
                                            self.weights[-1], right_weights[0],
                                            left_boundary, right_boundary + width_right,
                                            self.points[-1])
        else:
            # It's a normal, non bridge basis function, an 'element function':
            basis_function = ElementFunction(shapefunctions[-1], self.weights[-1],
                                             left_boundary, right_boundary)
        self.basis.append(basis_function)

    def valid(self, x_dense):
        """returns array of bools for which elements of x_dense are within the
        domain [self.left_boundary, self.right_boundary]"""
        valid = (self.left_boundary <= x_dense) & (x_dense <= self.right_boundary)
        return valid

    def make_vector(self, f):
        """Takes a function of space f, and returns an array containing the coefficients for that
        function's representation in this element's DVR basis."""
        from scipy.integrate import quad
        psi = np.zeros(self.N)
        for i, basis_function in enumerate(self.basis):
            projection, accuracy = quad(lambda x: f(x)*basis_function(x),
                                        basis_function.left_boundary,
                                        basis_function.right_boundary)
            psi[i] = projection
        return psi

    def interpolate_vector(self, psi, x_dense):
        """Takes a vector psi in the DVR basis and interpolates the spatial
        function it represents to the points in the array x_dense"""
        f = np.zeros(len(x_dense))
        for psi_i, basis_function in zip(psi, self.basis):
            f += psi_i*basis_function(x_dense)
        return f

    def differential_operator(self, order=1):
        """"Return a (self.N x self.N) array for the matrix representation of the derivative
        operator of a given order in the DVR basis."""
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
    element = Element(N, -2, 2)
    points = element.points
    weights = element.weights
    basis = element.basis
    # dn_dxn = dvr_basis.differential_operator(differentiation_order)

    # A dense grid for making plots
    x = np.linspace(element.left_boundary, element.right_boundary, 1000)
    dx = x[1] - x[0]

    # Our Gaussian wavefunction, its representation in the DVR basis,
    # and that representation's interpolation back onto the dense grid:

    def f(x):
        return np.exp(-x**2/4)

    psi_dense = f(x)
    psi = element.make_vector(f)
    psi_interpolated = element.interpolate_vector(psi, x)

    # The derivative of our Gaussian wavefunction, the derivative of its
    # representation in the DVR basis (computed with the DVR derivative
    # operator), and that derivatives interpolation back onto the dense grid:
    # d_psi_dense_dx = gradientn(psi_dense, dx, differentiation_order)
    # d_psi_dx = np.dot(dn_dxn, psi)
    # d_psi_dx_interpolated = dvr_basis.interpolate_vector(d_psi_dx, x_dense)

    # Plot the DVR basis functions:
    figure()
    subplot(211)
    title('DVR basis functions')
    for point, basis_function in zip(points, basis):
        plot(x, basis_function(x))
        plot(point, basis_function(point), 'ko')
    plot(points, np.zeros(len(points)), 'ko')
    grid(True)
    ylim(-1,6)

    # Plot the wavefunction and its interpolated DVR representation:
    subplot(223)
    title('Exact and DVR Gaussian')
    plot(x, psi_dense, 'b-')
    plot(x, psi_interpolated, 'r--')
    plot(points, psi/np.sqrt(weights), 'ko')
    grid(True)
    ylim(-0,1)

    # Plot the derivative of the wavefunction and its interpolated DVR representation:
    subplot(224)
    # title('Exact and DVR derivative')
    # plot(x_dense, d_psi_dense_dx, 'b-')
    # plot(x_dense, d_psi_dx_interpolated, 'r--')
    # plot(x, d_psi_dx/np.sqrt(w), 'ko')
    # grid(True)
    # ylim(-1,1)

    tight_layout()


def test_multiple_elements():
    N = 7
    n_elements = 3
    differentiation_order = 2

    # A dense grid for making plots
    x = np.linspace(-2,2,1000)
    dx = x[1] - x[0]
    boundaries = np.linspace(x.min(), x.max(), n_elements + 1, endpoint=True)

    elements = []
    for i, (x_l, x_r) in enumerate(zip(boundaries, boundaries[1:])):
        N_left = N if i > 0 else None
        N_right = N if i < n_elements - 1 else None
        element = Element(N, x_l, x_r, N_left, N_right, width_left=(x_r-x_l), width_right=(x_r-x_l))
        elements.append(element)

    # Our Gaussian wavefunction, its representation in the DVR basis,
    # and that representation's interpolation back onto the dense grid:
    def f(x):
        return np.exp(-x**2)

    psi_dense = f(x)

    psi = [e.make_vector(f) for e in elements]
    psi_interpolated = [e.interpolate_vector(psi_n, x) for psi_n, e in zip(psi, elements)]

    # Plot the FEDVR basis functions:
    figure()
    subplot(211)
    title('FEDVR basis functions')
    for element in elements:
        for point, weight, basis_function in zip(element.points, element.weights, element.basis):
            plot(x, basis_function(x))
            plot(point, 1/np.sqrt(weight), 'ko')
        plot(element.points, np.zeros(element.N), 'ko')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-1,7)

    # # Plot the derivatives of the FEDVR basis functions:
    # subplot(222)
    # title('derivative of FEDVR basis functions')
    # for element in elements:
    #     for x_i, u_i in zip(element.x, element.u):
    #         valid = element.valid(x_dense)
    #         plot(x_dense[valid], u_i.deriv()(x_dense[valid]))
    #         plot(x_i, u_i.deriv()(x_i), 'ko')
    #     plot(element.x, np.zeros(element.N), 'ko')
    # for boundary in boundaries:
    #     axvline(boundary, linestyle='--', color='k')
    # grid(True)
    # ylim(-200,200)

    # # Plot the second derivatives of the FEDVR basis functions:
    # subplot(224)
    # title('second derivative of FEDVR basis functions')
    # for element in elements:
    #     for x_i, u_i in zip(element.x, element.u):
    #         valid = element.valid(x_dense)
    #         plot(x_dense[valid], u_i.deriv(2)(x_dense[valid]))
    #         plot(x_i, u_i.deriv(2)(x_i), 'ko')
    #     plot(element.x, np.zeros(element.N), 'ko')
    # for boundary in boundaries:
    #     axvline(boundary, linestyle='--', color='k')
    # grid(True)
    # ylim(-2000,2000)

    # Plot the wavefunction and its interpolated FEDVR representation:
    subplot(212)
    title('Exact and FEDVR Gaussian')
    plot(x, psi_dense, 'b-')
    for element, psi_n, psi_interpolated_n in zip(elements, psi, psi_interpolated):
        plot(element.points, psi_n/np.sqrt(element.weights), 'ko')
        plot(element.points, np.zeros(element.N), 'ko')
        plot(x, psi_interpolated_n, '--')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-0,1)

    tight_layout()


if __name__ == '__main__':
    test_single_element()
    test_multiple_elements()
    show()
