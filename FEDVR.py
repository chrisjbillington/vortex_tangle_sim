from __future__ import division, print_function
from numpy.polynomial.polynomial import Polynomial
import numpy as np

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
    # Now we scale them from [-1, 1] to our domain:
    points = (points + 1)*(right_edge - left_edge)/2 + left_edge
    weights *= (right_edge - left_edge)/2
    return points, weights


def lobatto_shape_functions(points):
    """Returns a list of Polynomial objects representing the Lobatto
    shape functions for a set of Gauss-Lobatto quadrature points."""
    f = []
    for point in points:
        # The Lobatto shape function corresponding to a point is simply a
        # Lagrange basis polynomial with roots at all the other points:
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
        psi = np.zeros(self.N, dtype=complex)
        for i, (point, basis_function) in enumerate(zip(self.points, self.basis)):
            if not isinstance(basis_function, NullFunction):
                psi[i] = f(point)/basis_function(point)
        return psi

    def interpolate_vector(self, psi, x):
        """Takes a vector psi in the DVR basis and interpolates the spatial
        function it represents to the points in the array x.

        x may be larger than the domain of this element; the returned array
        will contain zeros at points outside the element's domain."""
        f = np.zeros(len(x), dtype=complex)
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
        operator operate on the bridge basis function in each element. If the
        two adjacent elements are identical, each matrix element is already
        half of the total, so no exchange is required."""
        d2_dx2 = np.zeros((self.N,self.N))
        for i, u_i in enumerate(self.basis):
            for j, u_j in enumerate(self.basis):
                # Evaluate the matrix element using the quadrature rule:
                for point, weight in zip(self.points, self.weights):
                    d2_dx2[i, j] += - weight * u_i.deriv()(point) * u_j.deriv()(point)
        return d2_dx2


class FiniteElements1D(object):
    """A class for operations on an array of identical DVR finite elements in one dimension"""
    def __init__(self, N, n_elements, left_boundary, right_boundary):
        self.N = N
        self.n_elements = n_elements
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        self.element_width = (right_boundary - left_boundary)/n_elements
        self.element_edges = np.linspace(left_boundary, right_boundary, n_elements + 1)

        # An element to represent all elements, since they are identical. We
        # instantiate it with the domain [0, element_width] and interpret its
        # points as being relative to the left side of the specific element
        # we're dealing with at any time.
        self.element = Element(N, 0, self.element_width, N_left=N, N_right=N,
                                        width_left=self.element_width, width_right=self.element_width)

        # construct a (self.n_elements x self.N) array for the quadrature points.
        # The following is an 'outer sum' between the position of points within an
        # element, and the position of the left edges of the elements:
        self.points = self.element.points + self.element_edges[:-1, np.newaxis]

        # The weights are identical in each element so we only need a length
        # self.N array:
        self.weights = self.element.weights

        # The values of each DVR basis function at its points:
        self.values = 1/np.sqrt(self.weights)
        # The basis functions at the edges have different normalisation, they
        # are 1/sqrt(2*w) rather than just 1/sqrt(w):
        self.values[0] = 1/np.sqrt(2*self.weights[0])
        self.values[-1] = 1/np.sqrt(2*self.weights[-1])

    def density_operator(self):
        """Returns a 1D array of size self.N representing the diagonals of the
        density operator rho on an element. vec.conj()*rho*vec then gives the
        wavefunction density |psi|^2 at each quadrature point."""
        rho = self.values**2
        return rho

    def derivative_operator(self):
        """Returns a (self.N x self.N) array for the derivative operator on
        each element"""
        return self.element.derivative_operator()

    def second_derivative_operator(self):
        """Returns a (self.N x self.N) array for the second derivative
        operator on each element"""
        return self.element.second_derivative_operator()

    def make_vector(self, f):
        """Takes a function of space f, and returns a (self.n_elements x
        self.N) array containing the coefficients for that function's
        representation in the DVR basis in each element."""
        psi = np.zeros(self.points.shape, dtype=complex)
        psi[:] = f(self.points)/self.values
        return psi

    def interpolate_vector(self, psi, npts):
        """Takes a (self.n_elements x self.N) array psi of coefficients in the
        FEDVR basis and interpolates the spatial function it represents to
        npts equally spaced points per element. Returns an array of the x
        points used and the values at those points."""
        # The array of points within an element:
        x = np.linspace(0, self.element_width, npts, endpoint=False)
        f = np.zeros((self.n_elements, npts), dtype=complex)
        for i, basis_function in enumerate(self.element.basis):
            f += psi[:, i, np.newaxis]*basis_function(x)

        # Create a 2D array for all the points in the domain:
        x_all = (x + self.element_edges[:-1, np.newaxis])
        # Flatten the output and include the rightmost boundary point:
        x_all = np.append(x_all.flatten(), self.points[-1,-1])
        f = np.append(f.flatten(), [f[0,0]])

        return x_all, f

    def get_values(self, psi):
        """Takes a (self.n_elements x self.N) array psi of DVR vectors in the
        DVR basis. Returns a 1D array of quadrature points and a 1D array of
        the values at those points of spatial function the DVR vector
        represents."""
        values = psi*self.values
        return self.points.flatten(), values.flatten()


class FiniteElements2D(object):
    """A class for operations on an array of identical DVR finite elements in two dimensions"""
    def __init__(self, n_elements_x, n_elements_y, Nx, Ny, n_components,
                 left_boundary, right_boundary,
                 bottom_boundary, top_boundary):
        self.n_elements_x = n_elements_x
        self.n_elements_y = n_elements_y
        self.Nx = Nx
        self.Ny = Ny
        self.n_components = n_components
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary

        self.element_width_x = (right_boundary - left_boundary)/n_elements_x
        self.element_edges_x = np.linspace(left_boundary, right_boundary, n_elements_x + 1)

        self.element_width_y = (top_boundary - bottom_boundary)/n_elements_y
        self.element_edges_y = np.linspace(bottom_boundary, top_boundary, n_elements_y + 1)

        # An element to represent all elements in the x dimension, since they
        # are identical. We instantiate it with the domain [0,
        # element_width_x] and interpret its points as being relative to the
        # left side of the specific element we're dealing with at any time.
        self.element_x = Element(Nx, 0, self.element_width_x, N_left=Nx, N_right=Nx,
                                 width_left=self.element_width_x, width_right=self.element_width_x)
        # The same for the y direction:
        self.element_y = Element(Ny, 0, self.element_width_y, N_left=Ny, N_right=Ny,
                                 width_left=self.element_width_y, width_right=self.element_width_y)

        # The shape of  vectors. The extra dimension at the end is so that
        # operators on one of our dimensions, which have size > 1 in that
        # dimension, have the same rank as vectors, so that we can say,
        # multiply them easily.
        self.shape = (self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)

        # construct a (n_elements_x, 1, Nx, 1, 1, 1) array for the quadrature points in
        # the x direction:
        self.points_x = self.element_x.points + self.element_edges_x[:-1, np.newaxis]
        self.points_x = self.points_x.reshape((n_elements_x, 1, Nx, 1, 1, 1))
        # The same for the y direction, shape (1, n_elements_y, 1, Ny, 1, 1):
        self.points_y = self.element_y.points + self.element_edges_y[:-1, np.newaxis]
        self.points_y = self.points_y.reshape((1, n_elements_y, 1, Ny, 1, 1))

        # The product of the weights over the 2D space; shape (1, 1, Nx, Ny, 1, 1):
        self.weights = np.outer(self.element_x.weights, self.element_y.weights)
        self.weights = self.weights.reshape((1, 1, Nx, Ny, 1, 1))

        # The values of each DVR basis function at its point; shape (1, 1, Nx, Ny, 1, 1):
        self.values = 1/np.sqrt(self.weights)
        # The basis functions at the edges have different normalisation, they
        # are 1/sqrt(2*w) rather than just 1/sqrt(w):
        self.values[:, :, 0, :] /= np.sqrt(2)
        self.values[:, :, -1, :] /= np.sqrt(2)
        self.values[:, :, :, 0] /= np.sqrt(2)
        self.values[:, :, :, -1] /= np.sqrt(2)

    def density_operator(self):
        """Returns a 1D array of size self.N representing the diagonals of the
        density operator rho on an element. vec.conj()*rho*vec then gives the
        wavefunction density |psi|^2 at each quadrature point."""
        rho = self.values**2
        # SCAFFOLDING: remove reshape:
        return rho.reshape((self.Nx, self.Ny))

    def derivative_operators(self):
        """Returns a (1, 1, Nx, Ny, 1, Nx) array for the first derivative
        operator on each element in the x direction, and a (1, 1, Nx, Ny, 1,
        Ny) array for the first derivative operator on each element in the y
        direction. The reason both have size Nx, Ny in the Nx and Ny
        dimensions is that the x first derivative operator is halved on the y
        edges of an element, and likewise for the y derivative operator. This
        is so that when we sum vectors at edges of elements, we get the right
        result. """
        gradx = self.element_x.second_derivative_operator()
        grady = self.element_y.second_derivative_operator()
        gradx = gradx.reshape(1, 1, self.Nx, 1, 1, self.Nx)
        grady = grady.reshape(1, 1, 1, self.Ny, 1, self.Ny)
        y_envelope = np.ones(self.Ny)
        y_envelope[0] = y_envelope[-1] = 0.5
        y_envelope.reshape((1, 1, 1, self.Ny, 1, 1))
        x_envelope = np.ones(self.Nx)
        x_envelope[0] = x_envelope[-1] = 0.5
        x_envelope = x_envelope.reshape((1, 1, self.Nx, 1, 1, 1))
        y_envelope = y_envelope.reshape((1, 1, 1, self.Ny, 1, 1))
        gradx = gradx * y_envelope
        grady = grady * x_envelope
        return gradx, grady

    def second_derivative_operators(self):
        """Returns a (1, 1, Nx, Ny, 1, Nx) array for the second derivative
        operator on each element in the x direction, and a (1, 1, Nx, Ny, 1,
        Ny) array for the second derivative operator on each element in the y
        direction. The reason both have size Nx, Ny in the Nx and Ny
        dimensions is that the x second derivative operator is halved on the y
        edges of an element, and likewise for the y derivative operator. This
        is so that when we sum vectors at edges of elements, we get the right
        result. """
        grad2x = self.element_x.second_derivative_operator()
        grad2y = self.element_y.second_derivative_operator()
        grad2x = grad2x.reshape(1, 1, self.Nx, 1, 1, self.Nx)
        grad2y = grad2y.reshape(1, 1, 1, self.Ny, 1, self.Ny)
        y_envelope = np.ones(self.Ny)
        y_envelope[0] = y_envelope[-1] = 0.5
        y_envelope.reshape((1, 1, 1, self.Ny, 1, 1))
        x_envelope = np.ones(self.Nx)
        x_envelope[0] = x_envelope[-1] = 0.5
        x_envelope = x_envelope.reshape((1, 1, self.Nx, 1, 1, 1))
        y_envelope = y_envelope.reshape((1, 1, 1, self.Ny, 1, 1))
        grad2x = grad2x * y_envelope
        grad2y = grad2y * x_envelope
        return grad2x, grad2y

    def interpolate_vector(self, psi, npts_x, npts_y):
        """Takes a (n_elements_x, n_elements_y, Nx, Ny, 1, 1)
        array psi of coefficients in the FEDVR basis and interpolates the
        spatial function it represents to npts equally spaced points per
        element. Returns arrays of the x and y points used and the values at
        those points."""
        # The array of points within an element:
        x = np.linspace(0, self.element_width_x, npts_x, endpoint=False)
        y = np.linspace(0, self.element_width_y, npts_y, endpoint=False)
        f = np.zeros((self.n_elements_x, self.n_elements_y, npts_x, npts_y), dtype=complex)
        for j, basis_function_x in enumerate(self.element_x.basis):
            for k, basis_function_y in enumerate(self.element_y.basis):
                f += (psi[:, :, j, k, 0, 0, np.newaxis, np.newaxis] * # SCAFFOLDING: remove 0, 0,
                      basis_function_x(x[:, np.newaxis]) *
                      basis_function_y(y[np.newaxis, :]))

        x_all = (x + self.element_edges_x[:-1, np.newaxis]).flatten()
        y_all = (y + self.element_edges_y[:-1, np.newaxis]).flatten()
        shape = (self.n_elements_x * npts_x, self.n_elements_y * npts_y)
        f_reshaped = f.transpose(0, 2, 1, 3).reshape(shape)

        return x_all, y_all, f_reshaped

    def make_vector(self, f):
        """Takes a function of space f, and returns a (self.n_elements_x x
        self.n_elements_y x self.Nx x self.Ny) array containing the coefficients
        for that function's representation in the DVR basis in each
        element."""
        psi = np.zeros(self.shape, dtype=complex)
        psi[:] = f(self.points_x, self.points_y)/self.values
        return psi

    def get_values(self, psi):
        """Takes a (self.n_elements_x x self.n_elements_y x self.Nx x self.Ny)
        array psi of DVR coefficients in the FEDVR basis. Returns two 2D arrays of
        quadrature points and a 2D array of the values at those points of the
        spatial function the DVR vector represents."""
        values = psi*self.values
        output_shape = (self.n_elements_x*self.Nx, self.n_elements_y*self.Ny)
        x = self.points_x.flatten()
        y = self.points_y.flatten()
        values = values.transpose(0,2,1,3).reshape(output_shape)
        return x, y, values
