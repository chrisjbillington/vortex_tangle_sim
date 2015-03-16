from __future__ import division, print_function
from pylab import figure, plot, show, grid, subplot, ylim, title, axvline, tight_layout, imshow, gca, clf
from numpy.polynomial.polynomial import Polynomial
from matplotlib import ticker
import numpy as np
import pylab as pl

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
    def __init__(self, point, weight, shapefunction, element_left_boundary, element_right_boundary):
        self.point = point
        self.polynomial = shapefunction/np.sqrt(weight)
        self.left_boundary = element_left_boundary
        self.right_boundary = element_right_boundary
        self.weight = weight

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

    def get_projection(self, f):
        """Return the projection of the function f onto this DVR basis function"""
        return f(self.point)/self(self.point)

    def derivative(self, arg):
        result = self.polynomial.deriv()(arg)
        valid = (self.left_boundary <= arg) & (arg <= self.right_boundary)
        if isinstance(arg, np.ndarray):
            result[~valid] = 0
        elif not valid:
            result = 0
        return result

    def second_derivative(self, arg):
        result = self.polynomial.deriv(2)(arg)
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
    def __init__(self, point, left_weight, right_weight,
                 left_shapefunction, right_shapefunction,
                 left_element_left_boundary, right_element_right_boundary):
        self.point = point
        self.left_segment = left_shapefunction/np.sqrt(left_weight + right_weight)
        self.right_segment = right_shapefunction/np.sqrt(left_weight + right_weight)
        self.left_boundary = left_element_left_boundary
        self.right_boundary = right_element_right_boundary
        self.left_weight = left_weight
        self.right_weight = right_weight

    def __call__(self, arg):
        left_valid = (self.left_boundary <= arg) & (arg < self.point)
        right_valid = (arg >= self.point) & (arg <= self.right_boundary)
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

    def get_projection(self, f):
        """Return the projection of the function f onto this DVR basis function"""
        return f(self.point)/self(self.point)

    def left_derivative(self, arg):
        return self.left_segment.deriv()(arg)

    def right_derivative(self, arg):
        return self.right_segment.deriv()(arg)

    def left_second_derivative(self, arg):
        return self.left_segment.deriv(2)(arg)

    def right_second_derivative(self, arg):
        return self.right_segment.deriv(2)(arg)


class NullFunction(object):
    """A stand in for an ElementFunction that can be used at the edges of the
    domain in order to impose zero boundary conditions."""
    def __call__(self, arg):
        if isinstance(arg, np.ndarray):
            return np.zeros(arg.shape)
        else:
            return 0

    def derivative(self, arg):
        return np.zeros_like(arg)

    def second_derivative(self, arg):
        return np.zeros_like(arg)

    def get_projection(self, f):
        return 0


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
            # Then the leftmost basis function is a bridge function. We need
            # to know the quadrature points, weights, and shape functions in
            # the element to the left of us in order to construct the bridge
            # function:
            left_points, left_weights = gauss_lobatto_points_and_weights(
                                            N_left, left_boundary - width_left, left_boundary)
            left_shapefunctions = lobatto_shape_functions(left_points)
            first_basis_function = BridgeFunction(self.points[0], left_weights[-1], self.weights[0],
                                                  left_shapefunctions[-1], shapefunctions[0],
                                                  left_boundary - width_left, right_boundary)
        else:
            # We use this to impose zero boundary conditions:
            first_basis_function = NullFunction()
        self.basis.append(first_basis_function)

        # Now all the internal basis functions:
        for point, weight, shapefunction in zip(self.points, self.weights, shapefunctions)[1:-1]:
            basis_function = ElementFunction(point, weight, shapefunction, left_boundary, right_boundary)
            self.basis.append(basis_function)

        # Now the rightmost basis function:
        if N_right is not None:
            # Then the leftmost basis function is a bridge function. We need
            # to know the quadrature points, weights, and shape functions in
            # the element to the right of us in order to construct the bridge
            # function:
            right_points, right_weights = gauss_lobatto_points_and_weights(
                                              N_right, right_boundary, right_boundary + width_right)
            right_shapefunctions = lobatto_shape_functions(right_points)
            last_basis_function = BridgeFunction(self.points[-1], self.weights[-1], right_weights[0],
                                                 shapefunctions[-1], right_shapefunctions[0],
                                                 left_boundary, right_boundary + width_right)
        else:
            # We use this to impose zero boundary conditions:
            last_basis_function = NullFunction()
        self.basis.append(last_basis_function)

    def valid(self, x):
        """returns array of bools for which elements of x are within the
        domain [self.left_boundary, self.right_boundary]"""
        valid = (self.left_boundary <= x) & (x <= self.right_boundary)
        return valid

    def make_vector(self, f):
        """Takes a function of space f, and returns an array containing the
        coefficients for that function's representation in this element's DVR
        basis."""
        psi = np.zeros(self.N)
        for i, basis_function in enumerate(self.basis):
            psi[i] = basis_function.get_projection(f)
        return psi

    def interpolate_vector(self, psi, x):
        """Takes a vector psi in the DVR basis and interpolates the spatial
        function it represents to the points in the array x"""
        f = np.zeros(len(x))
        for i, (psi_i, basis_function) in enumerate(zip(psi, self.basis)):
            f += psi_i*basis_function(x)
        return f

    def derivative_operator(self):
        """"Return a (self.N x self.N) array for the matrix representation of
        the derivative operator in the DVR basis."""
        d_dx = np.zeros((self.N,self.N))
        for i, (x_i, w_i, u_i) in enumerate(zip(self.points, self.weights, self.basis)):
            for j, u_j in enumerate(self.basis):
                if i == j:
                    # Diagonals of derivative operator are zero, including bridge functions:
                    d_dx[i, j] = 0
                    continue
                if isinstance(u_j, BridgeFunction):
                    # We only need the interior segments of the bridge
                    # functions to evaluate the matrix element of non bridge
                    # functions with them:
                    if j == 0:
                        # It's a bridge function on the left of the element.
                        # Only its right segment is interior to the element:
                        du_j_dx = u_j.right_derivative
                    elif j == self.N-1:
                        # It's a bridge function on the right of the element.
                        # Only its left segment is interior to the element:
                        du_j_dx = u_j.left_derivative
                else:
                    # Not a bridge function:
                    du_j_dx = u_j.derivative
                # Evaluate the matrix element using the quadrature rule; the
                # sum for which has only one nonzero term:
                d_dx[i, j] = w_i * u_i(x_i) * du_j_dx(x_i)
        return d_dx

    def second_derivative_operator(self):
        """"Return a (self.N x self.N) array for the matrix representation of
        the derivative operator in the DVR basis."""
        d2_dx2 = np.zeros((self.N,self.N))
        for i, (x_i, w_i, u_i) in enumerate(zip(self.points, self.weights, self.basis)):
            for j, u_j in enumerate(self.basis):
                if i == j and isinstance(u_i, BridgeFunction):
                    # Matrix element of a bridge function with itself is the
                    # sum of the matrix elements of each segment with itself,
                    # evaluated with the quadrature rules on either side of
                    # the boundary (possibly having different weights), plus a
                    # finite contribution due to the discontinuous first
                    # derivative across the boundary. The integral of the
                    # derivative of a step function must give back the step
                    # function, so we find that the integral of the second
                    # derivative across the boundary is simply equal to the
                    # size of the discontinuous jump in the first derivative
                    # at that point. If you like you can think of the second
                    # derivative having a dirac delta at the boundary.

                    d2_dx2[i, j] = (u_i.left_weight * u_i(x_i) * u_i.left_second_derivative(x_i) +
                                    u_i.right_weight * u_i(x_i) * u_i.right_second_derivative(x_i) +
                                    u_i.right_derivative(x_i) - u_i.left_derivative(x_i))
                    continue
                if isinstance(u_j, BridgeFunction):
                    # We only need the interior segments of the bridge
                    # functions to evaluate the matrix element of non bridge
                    # functions with them:
                    if j == 0:
                        # It's a bridge function on the left of the element.
                        # Only its right segment is interior to the element:
                        d2u_j_dx2 = u_j.right_second_derivative
                    elif j == self.N-1:
                        # It's a bridge function on the right of the element.
                        # Only its left segment is interior to the element:
                        d2u_j_dx2 = u_j.left_second_derivative
                else:
                    # Not a bridge function:
                    d2u_j_dx2 = u_j.second_derivative
                # Evaluate the matrix element using the quadrature rule; the
                # sum for which has only one nonzero term:
                d2_dx2[i, j] = w_i * u_i(x_i) * d2u_j_dx2(x_i)
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

    def differential_operators(self, order=1):
        """"Return a list of (self.N x self.N) arrays for the matrix representations of the derivative
        operators of a given order in the DVR basis of each element."""
        differential_operators = []
        for element in self.elements:
            differential_operator = element.differential_operator(order)
            differential_operators.append(differential_operator)
        return differential_operators


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


def test_second_derivative():
    # A dense grid for making plots
    x = np.linspace(-2,2,100000)
    dx = x[1] - x[0]
    # boundaries = np.array([-2,-1,-0.5,-0.25,0,1,2])
    boundaries = np.linspace(-2,2,7, endpoint=True)
    widths = np.diff(boundaries)
    # N = [7,8,9,10,7,6]
    N = np.zeros(len(widths)) + 7
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
    # d_psi_dense_dx = gradientn(psi_dense, dx, 1)
    d_psi_dense_dx = gradientn(psi_dense, dx, 2)

    # Plot the derivative of the wavefunction using the total derivative operator:
    subplot(121)
    title('Total derivative operator')

    # Lets construct the overall derivative operator:
    total_N = sum(N) - n_elements + 1
    D_total = np.zeros((total_N, total_N))
    psi_total = np.zeros(total_N)
    weights = np.zeros(total_N)
    points = np.zeros(total_N)
    start_index = 0
    for i, element in enumerate(elements):
        dn_dxn = element.derivative_operator()
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
    imshow(D_plot, interpolation='nearest')
    loc = ticker.IndexLocator(base=1, offset=0)
    ax = gca()
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.grid(which='minor', axis='both', linestyle='-')
    pl.colorbar()

    subplot(122)
    # clf()
    title('Exact and FEDVR derivative with total operator')
    plot(x, d_psi_dense_dx, 'k-')
    d_psi_total_dx = np.dot(D_total, psi_total)
    plot(points, weights*d_psi_total_dx, 'ko')
    for boundary in boundaries:
        axvline(boundary, linestyle='--', color='k')
    grid(True)
    ylim(-10,4)
    return D_total



if __name__ == '__main__':
    # test_single_element()
    # test_multiple_elements()
    # show()
    test_second_derivative()
    show()
