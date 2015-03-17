from __future__ import division, print_function
from pylab import figure, plot, show, grid, subplot, ylim, title, axvline, tight_layout, imshow, gca, clf
from numpy.polynomial.polynomial import Polynomial
from matplotlib import ticker
import numpy as np
import pylab as pl

def gauss_lobatto_points_and_weights(N, left_edge, right_edge):
    """Returns the spatial points and weights for the N-point Gauss-Lobatto
    quadrature, scaled for the spatial interval [left_edge, right_edge]."""
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
    points = (points + 1)*(right_edge - left_edge)/2 + left_edge
    weights *= (right_edge - left_edge)/2
    return points, weights


def lobatto_shape_functions(points):
    """Returns a list of Polynomial objects representing the Lobatto
    shape functions for a set of Gauss-Lobatto quadrature points."""
    f = []
    for point in points:
        # The Lobatto shape function corresponding to a point is simply the
        # Lagrange basis polynomial with all the other points as roots:
        other_points = points[points!=point]
        f_i = Polynomial.fromroots(other_points)/np.prod(point - other_points)
        f.append(f_i)
    return f


class NullFunction(Polynomial):
    """A Polynomial object that is zero everywhere. Useful as a stand-in for
    basis functions at the edge of a region when zero boundary conditions are
    desired"""
    def __init__(self):
        Polynomial.__init__(self, [0])

    def deriv(self, order=1):
        return self


class Element(object):
    """A class for operations with the N-point discrete variable
    representation basis for Gauss-Lobatto quadrature on an interval
    [left_edge, right_edge]. If this DVR basis is just one element of
    many finite elements, then N_left and N_right specify how many DVR basis
    functions the elements to the left and right of this one have, and
    width_left and width_right specify the widths of those elements. This is
    important for normalising the basis functions at the edges. If N_left or
    N_right is None, that means that that edge corresponds to a boundary of
    the problem, at which zero boundary conditions are imposed.

    On the edge joining two adjacent Elements, each Element object has a basis
    function representing only its segment of the joining bridge function. In
    order to construct the total second derivative operator, therefore, the
    matrix elements produced on either side of the bridge must be summed to
    produce the total matrix element for that bridge function.

    The first derivative operator has zeros on the diagonal, so no such
    communication is required to combine first derivative matrix elements
    either size of a bridge.

    Attributes:
        N:              number of quadrature points/basis functions

        left_edge:      left edge of element

        right_edge:     right edge of element

        points:         quadrature points

        weights:        quadrature weights

        basis:          DVR basis polynomials, valid in in the interval
                        [left_edge, right_edge]. The left and rightmost basis
                        functions are either a segment of bridge function, or
                        a NullFunction if they are on a boundary.
    """
    def __init__(self, N, left_edge=-1, right_edge=1,
                 N_left=None, N_right=None, width_left=2, width_right=2):
        self.N = N
        self.left_edge = left_edge
        self.right_edge = right_edge

        # Construct our DVR basis functions, which are Lobatto shape functions
        # normalised according to the norm defined by quadrature integration:
        self.points, self.weights = gauss_lobatto_points_and_weights(N, left_edge, right_edge)
        shapefunctions = lobatto_shape_functions(self.points)
        self.basis = []

        # Is the leftmost basis function a boundary or a bridge?
        if N_left is None:
            leftmost_basis_function = NullFunction()
        else:
            _, left_weights = gauss_lobatto_points_and_weights(N_left, left_edge - width_left, left_edge)
            leftmost_basis_function = shapefunctions[0]/np.sqrt(left_weights[-1] + self.weights[0])
        self.basis.append(leftmost_basis_function)

        # Now all the internal basis functions:
        for point, weight, shapefunction in zip(self.points, self.weights, shapefunctions)[1:-1]:
            basis_function = shapefunction/np.sqrt(weight)
            self.basis.append(basis_function)

        # Is the rightmost basis function a boundary or a bridge?
        if N_right is None:
            rightmost_basis_function = NullFunction()
        else:
            _, right_weights = gauss_lobatto_points_and_weights(
                                   N_right, right_edge, right_edge + width_right)
            rightmost_basis_function = shapefunctions[-1]/np.sqrt(self.weights[-1] + right_weights[0])
        self.basis.append(rightmost_basis_function)

    def valid(self, x):
        """Returns array of bools for whether x is within the element's
        domain"""
        return (self.left_edge <= x) & (x <= self.right_edge)

    def make_vector(self, f):
        """Takes a function of space f, and returns an array containing the
        coefficients for that function's representation in this element's DVR
        basis.

        If there is a bridge at either edge of the element, then the
        coefficient returned for it is that of the whole bridge function, not
        just our element's segment. So no summing or anything is required
        across bridges; to within rounding error, make_vector(f) called on two
        adjacent elements will return the same coefficient for the point
        joining them.

        Coefficients are defined to be zero at the boundary of the problem.
        For sensible results, f should be zero there too."""
        psi = np.zeros(self.N)
        for i, (point, basis_function) in enumerate(zip(self.points, self.basis)):
            if not isinstance(basis_function, NullFunction):
                psi[i] = f(point)/basis_function(point)
        return psi

    def interpolate_vector(self, psi, x):
        """Takes a vector psi in the DVR basis and interpolates the spatial
        function it represents to the points in the array x.

        x may be larger than the domain of this element; the returned array
        will contain zeros at points outside the element's domain."""
        f = np.zeros(len(x))
        valid = self.valid(x)
        for i, (psi_i, basis_function) in enumerate(zip(psi, self.basis)):
            f[valid] += psi_i*basis_function(x[valid])
        return f

    def derivative_operator(self):
        """"Return a (self.N x self.N) array for the matrix representation of
        the derivative operator in the DVR basis.

        Its diagonals are zero, and so the first and last diagonals can be
        considered to be the matrix elements for the total bridge function, if
        any, with no communication between elements required. """
        d_dx = np.zeros((self.N,self.N))
        for i, (x_i, w_i, u_i) in enumerate(zip(self.points, self.weights, self.basis)):
            for j, u_j in enumerate(self.basis):
                if i == j:
                    # Diagonals of derivative operator are zero:
                    d_dx[i, j] = 0
                else:
                    # Evaluate the matrix element using the quadrature rule;
                    # the sum for which has only one nonzero term:
                    d_dx[i, j] = w_i * u_i(x_i) * u_j.deriv()(x_i)
        return d_dx

    def second_derivative_operator(self):
        """"Return a (self.N x self.N) array for the matrix representation of
        the second derivative operator in the DVR basis.

        The first and last diagonals correspond to the matrix elements for
        only our element's segment of any the bridge functions, and so must be
        summed with the matrix element for the other segment in order to
        obtain the total second derivative matrix element for the bridge
        function. In actual implementation, more likely one will sum them and
        then divide by two, in order to have half of the kinetic energy
        operator operate on the bridge basis function in each element"""
        d2_dx2 = np.zeros((self.N,self.N))
        for i, u_i in enumerate(self.basis):
            for j, u_j in enumerate(self.basis):
                # Evaluate the matrix element using the quadrature rule:
                for point, weight in zip(self.points, self.weights):
                    d2_dx2[i, j] += - weight * u_i.deriv()(point) * u_j.deriv()(point)
        return d2_dx2


class FiniteElements(object):
    """Convenience class for operations on an even grid of elements"""
    def __init__(self, N, n_elements, left_boundary, right_boundary):
        self.N = N
        self.n_elements = n_elements
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.boundaries = np.linspace(left_boundary, right_boundary, n_elements+1, endpoint=True)
        width = self.boundaries[1] - self.boundaries[0]
        self.elements = []
        for i, (left, right) in enumerate(zip(self.boundaries, self.boundaries[1:])):
            N_left = N if i else None
            N_right = N if i < n_elements - 1 else None
            element = Element(N, left, right, N_left, N_right, width, width)
            self.elements.append(element)

    def make_vectors(self, f):
        """Takes a function of space f, and returns a list of arrays
        containing the coefficients for that function's representation in each
        element's DVR basis."""
        vectors = []
        for element in self.elements:
            vector = element.make_vector(f)
            vectors.append(vector)
        return vectors

    def interpolate_vectors(self, vectors, x):
        """Takes a list of vectors in the DVR basis of each element and
        interpolates the spatial function it represents to the points in the
        array x"""
        psi_interpolated = np.zeros(len(x))
        points = []
        values = []
        for vector, element in zip(vectors, self.elements):
            valid = element.valid(x)
            psi_interpolated[valid] = element.interpolate_vector(vector, x[valid])
            for i, (point, basis_function) in enumerate(zip(element.points, element.basis)):
                points.append(point)
                values.append(vector[i]*basis_function(point))
        return psi_interpolated, np.array(points), np.array(values)

    def derivative_operators(self):
        """"Return a list of (self.N x self.N) arrays for the matrix
        representations of the derivative operators of a given order in
        the DVR basis of each element."""
        operators = []
        for element in self.elements:
            operator = element.derivative_operator()
            operators.append(operator)
        return operators

    def second_derivative_operators(self):
        """"Return a list of (self.N x self.N) arrays for the matrix
        representations of the second derivative operators of a given order in
        the DVR basis of each element."""
        operators = []
        for element in self.elements:
            operator = element.second_derivative_operator()
            operators.append(operator)
        return operators


def gradientn(y, dx, n=1):
    """Returns the nth derivative of y using repeated applications of
    np.gradient. Values near the endpoints may be very wrong."""
    result = y
    for i in range(n):
        result = np.gradient(result, dx)
    return result


def test_single_element():
    N = 9

    # Get our quadrature points and weights, our DVR basis functions, and
    # our differential operator in the DVR basis:
    element = Element(N, -2, 2)
    points = element.points
    weights = element.weights
    basis = element.basis
    dn_dxn = element.derivative_operator()

    # A dense grid for making plots
    x = np.linspace(element.left_boundary, element.right_boundary, 1000)
    dx = x[1] - x[0]

    # Our Gaussian wavefunction, its representation in the DVR basis,
    # and that representation's interpolation back onto the dense grid:

    def f(x):
        return np.exp(-x**2)

    psi_dense = f(x)
    psi = element.make_vector(f)
    psi_interpolated = element.interpolate_vector(psi, x)

    # The derivative of our Gaussian wavefunction, the derivative of its
    # representation in the DVR basis (computed with the DVR derivative
    # operator), and that derivatives interpolation back onto the dense grid:
    d_psi_dense_dx = gradientn(psi_dense, dx, 1)
    d_psi_dx = np.dot(dn_dxn, psi)
    d_psi_dx_interpolated = element.interpolate_vector(d_psi_dx, x)

    # Plot the DVR basis functions:
    figure()
    subplot(311)
    title('DVR basis functions')
    for point, basis_function in zip(points, basis):
        plot(x, basis_function(x))
        plot(point, basis_function(point), 'ko')
    plot(points, np.zeros(len(points)), 'ko')
    grid(True)
    ylim(-1,6)

    # Plot the wavefunction and its interpolated DVR representation:
    subplot(312)
    title('Exact and DVR Gaussian')
    plot(x, psi_dense, 'b-')
    plot(x, psi_interpolated, 'r--')
    plot(points, psi/np.sqrt(weights), 'ko')
    grid(True)
    ylim(-0,1)

    # Plot the derivative of the wavefunction and its interpolated DVR representation:
    subplot(313)
    title('Exact and DVR derivative')
    plot(x, d_psi_dense_dx, 'b-')
    plot(x, d_psi_dx_interpolated, 'r--')
    plot(points, d_psi_dx/np.sqrt(weights), 'ko')
    grid(True)
    ylim(-1,1)

    tight_layout()


def test_multiple_elements():
    # A dense grid for making plots
    x = np.linspace(-2,2,100000)
    dx = x[1] - x[0]
    boundaries = np.array([-2,-1,-0.5,-0.25,0,1,2])
    # boundaries = np.linspace(-2,2,7, endpoint=True)
    widths = np.diff(boundaries)
    N = [7,8,9,10,7,6]
    # N = np.zeros(len(widths)) + 7
    n_elements = len(boundaries) - 1

    elements = []
    for i, (x_l, x_r) in enumerate(zip(boundaries, boundaries[1:])):
        N_left = N[i-1] if i > 0 else None
        N_right = N[i+1] if i < n_elements - 1 else None
        width_left = widths[i-1] if i > 0 else None
        width_right = widths[i+1] if i < n_elements - 1 else None
        element = Element(N[i], x_l, x_r, N_left, N_right, width_left=width_left, width_right=width_right)
        elements.append(element)

    # Our Gaussian wavefunction, its representation in the DVR basis,
    # and that representation's interpolation back onto the dense grid:
    def f(x):
        return np.exp(-x**2/(0.5**2))

    psi_dense = f(x)
    psi = [e.make_vector(f) for e in elements]
    psi_interpolated = [e.interpolate_vector(psi_n, x) for psi_n, e in zip(psi, elements)]

    # The exact derivative of the wavefunction:
    # d_psi_dense_dx = gradientn(psi_dense, dx, 1)
    d_psi_dense_dx = gradientn(psi_dense, dx, 2)

    # Plot the FEDVR basis functions:
    figure()
    subplot(231)
    title('FEDVR basis functions')
    for element in elements:
        for point, weight, basis_function in zip(element.points, element.weights, element.basis):
            plot(x, basis_function(x))
            plot(point, 1*basis_function(point), 'ko')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-2,10)

    # Plot the wavefunction and its interpolated FEDVR representation:
    subplot(232)
    title('Exact and FEDVR Gaussian')
    plot(x, psi_dense, 'k-')
    for element, psi_n, psi_interpolated_n in zip(elements, psi, psi_interpolated):
        valid = element.valid(x)
        for point, basis_function, c in zip(element.points, element.basis, psi_n):
            plot(point, c*basis_function(point), 'ko')
        plot(x[valid], psi_interpolated_n[valid], '--')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-0,1)

    # Plot the derivative of the wavefunction:
    subplot(233)
    title('Exact and FEDVR elementwise derivative')
    plot(x, d_psi_dense_dx, 'k-')
    d_psi_dx = []
    for element, psi_n in zip(elements, psi):
        # dn_dxn = element.derivative_operator()
        dn_dxn = element.second_derivative_operator()
        d_psi_n_dx = np.dot(dn_dxn, psi_n)
        d_psi_dx.append(d_psi_n_dx)
    for left_deriv, right_deriv in zip(d_psi_dx, d_psi_dx[1:]):
        left_deriv[-1] = right_deriv[0] = left_deriv[-1] + right_deriv[0]

    for element, psi_n , d_psi_n_dx in zip(elements, psi, d_psi_dx):
        valid = element.valid(x)
        d_psi_dx_interpolated = element.interpolate_vector(d_psi_n_dx, x)
        for i, point in enumerate(element.points):
            plot(point, d_psi_n_dx[i]*element.basis[i](point), 'ko')
        plot(x[valid], d_psi_dx_interpolated[valid], '--')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-10, 4)

    # Plot the derivative of the wavefunction using the total derivative operator:
    subplot(234)
    title('Total derivative operator')

    # Lets construct the overall derivative operator:
    total_N = sum(N) - n_elements + 1
    D_total = np.zeros((total_N, total_N))
    psi_total = np.zeros(total_N)
    weights = np.zeros(total_N)
    points = np.zeros(total_N)
    start_index = 0
    for i, element in enumerate(elements):
        # dn_dxn = element.derivative_operator()
        dn_dxn = element.second_derivative_operator()
        end_index = start_index + N[i]
        D_total[start_index:end_index, start_index:end_index] = dn_dxn
        psi_total[start_index:end_index] = psi[i]
        for j, (point, basis_function) in enumerate(zip(element.points, element.basis)):
            weights[start_index+j] = basis_function(point)
            points[start_index+j] = point
        start_index += N[i] - 1

    D_plot = D_total.copy()
    D_plot[np.abs(D_plot)<1e-8] = np.nan
    # D_plot = np.sign(D_plot)*np.log(np.abs(D_plot))
    imshow(D_plot, interpolation='nearest')
    loc = ticker.IndexLocator(base=1, offset=0)
    ax = gca()
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.grid(which='minor', axis='both', linestyle='-')

    subplot(235)
    # clf()
    title('Exact and FEDVR derivative with total operator')
    plot(x, d_psi_dense_dx, 'k-')
    d_psi_total_dx = weights*np.dot(D_total, psi_total)
    plot(points, d_psi_total_dx, 'ko')
    for element, psi_n , d_psi_n_dx in zip(elements, psi, d_psi_dx):
        valid = element.valid(x)
        d_psi_dx_interpolated = element.interpolate_vector(d_psi_n_dx, x)
        for i, point in enumerate(element.points):
            plot(point, d_psi_n_dx[i]*element.basis[i](point), 'ko')
        plot(x[valid], d_psi_dx_interpolated[valid], '--')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-10, 4)


def test_derivative(order=1, equal=True):
    figure()
    # A dense grid for making plots
    x = np.linspace(-2,2,100000)
    dx = x[1] - x[0]
    if equal:
        boundaries = np.linspace(-2, 2, 12)
    else:
        boundaries = np.array([-2,-1,-0.5,-0.25,0,1,2])
    widths = np.diff(boundaries)
    if equal:
        N = np.zeros(len(widths)) + 7
    else:
        N = [7,8,9,10,12,6]
    n_elements = len(boundaries) - 1

    elements = []
    for i, (x_l, x_r) in enumerate(zip(boundaries, boundaries[1:])):
        N_left = N[i-1] if i > 0 else None
        N_right = N[i+1] if i < n_elements - 1 else None
        width_left = widths[i-1] if i > 0 else None
        width_right = widths[i+1] if i < n_elements - 1 else None
        element = Element(N[i], x_l, x_r, N_left, N_right, width_left=width_left, width_right=width_right)
        elements.append(element)

    # Our Gaussian wavefunction, its representation in the DVR basis,
    # and that representation's interpolation back onto the dense grid:
    def f(x):
        return np.exp(-x**2/(0.5**2))

    psi_dense = f(x)
    psi = [e.make_vector(f) for e in elements]

    # The exact derivative of the wavefunction:
    d_psi_dense_dx = gradientn(psi_dense, dx, order)

    # Plot the derivative of the wavefunction using the total derivative operator:
    subplot(121)
    if order == 1:
        title('First derivative operator')
    else:
        title('Second derivative operator')

    # Lets construct the overall derivative operator:
    total_N = sum(N) - n_elements + 1
    D_total = np.zeros((total_N, total_N))
    psi_total = np.zeros(total_N)
    weights = np.zeros(total_N)
    points = np.zeros(total_N)
    start_index = 0
    for i, element in enumerate(elements):
        dn_dxn = element.derivative_operator()
        if order == 2:
            dn_dxn = element.second_derivative_operator()
        else:
            dn_dxn = element.derivative_operator()
        end_index = start_index + N[i]
        D_total[start_index:end_index, start_index:end_index] += dn_dxn
        psi_total[start_index:end_index] = psi[i]
        for j, (point, basis_function) in enumerate(zip(element.points, element.basis)):
            weights[start_index+j] = basis_function(point)
            points[start_index+j] = point
        start_index += N[i] - 1

    D_plot = D_total.copy()
    D_plot[np.abs(D_plot)<1e-8] = np.nan
    imshow(D_plot, interpolation='nearest')
    loc = ticker.IndexLocator(base=1, offset=0)
    ax = gca()
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.grid(which='minor', axis='both', linestyle='-')
    pl.colorbar()

    subplot(122)
    if order == 1:
        title('Exact and FEDVR first derivative')
    else:
        title('Exact and FEDVR second derivative')
    plot(x, d_psi_dense_dx, 'k-')
    d_psi_total_dx = np.dot(D_total, psi_total)
    plot(points, weights*d_psi_total_dx, 'ko')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-10,4)


if __name__ == '__main__':
    # test_single_element()
    # test_multiple_elements()
    test_derivative(order=1)
    test_derivative(order=2)
    show()
