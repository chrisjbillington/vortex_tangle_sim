from __future__ import division, print_function
import os
import sys
import time
import traceback

import numpy as np
from mpi4py import MPI
import h5py

from FEDVR import FiniteElements2D

def show_slices(slices, shape, show=False):
    """Function that shows what points are selected by a particular slice
    operation"""
    import pylab as pl
    from matplotlib import ticker
    pl.figure()
    n_elements_x, n_elements_y, Nx, Ny = shape
    psi = np.zeros(shape)
    psi[slices] = 1
    psi = psi.transpose(0, 2, 1, 3).reshape((n_elements_x * Nx, n_elements_y * Ny))
    pl.imshow(psi.transpose(), origin='lower', interpolation='nearest', cmap='gray_r')
    locx = ticker.IndexLocator(base=Nx, offset=0)
    locy = ticker.IndexLocator(base=Ny, offset=0)
    ax = pl.gca()
    ax.xaxis.set_minor_locator(locx)
    ax.yaxis.set_minor_locator(locy)
    ax.grid(which='minor', axis='both', linestyle='-')
    if show:
        pl.show()

# Some slice objects for conveniently slicing axes in multidimensional arrays:
FIRST = np.s_[:1]
LAST = np.s_[-1:]
ALL = np.s_[:]
INTERIOR = np.s_[1:-1]
ALL_BUT_FIRST = np.s_[1:]
ALL_BUT_LAST = np.s_[:-1]

# Each of the following is (x_elements, y_elements, x_points, y_points)
ALL_ELEMENTS_AND_POINTS = (ALL, ALL, ALL, ALL)
LEFT_BOUNDARY = (FIRST, ALL, FIRST, ALL) # The points on the left edge of the left boundary element
RIGHT_BOUNDARY = (LAST, ALL, LAST, ALL) # The points on the right edge of the right boundary element
BOTTOM_BOUNDARY = (ALL, FIRST, ALL, FIRST) # The points on the bottom edge of the bottom boundary element
TOP_BOUNDARY = ( ALL, LAST, ALL, LAST) # The points on the top edge of the top boundary element
INTERIOR_ELEMENTS = (INTERIOR, INTERIOR, ALL, ALL) # All points in all elements that are not boundary elements
INTERIOR_POINTS = (ALL, ALL, INTERIOR, INTERIOR) # Points that are not edgepoints in all elements
LEFT_INTERIOR_EDGEPOINTS = (ALL_BUT_FIRST, ALL, FIRST, ALL) # All points on the left edge of a non-border element
RIGHT_INTERIOR_EDGEPOINTS = (ALL_BUT_LAST, ALL, LAST, ALL) # All points on the right edge of a non-border element
BOTTOM_INTERIOR_EDGEPOINTS = (ALL, ALL_BUT_FIRST, ALL, FIRST) # All points on the bottom edge of a non-border element
TOP_INTERIOR_EDGEPOINTS = (ALL, ALL_BUT_LAST, ALL, LAST) # All points on the top edge of a non-border element
# All but the last points in the x and y directions, so that we don't double count them when summing.
DONT_DOUBLE_COUNT_EDGES = (ALL, ALL, ALL_BUT_LAST, ALL_BUT_LAST)

def get_factors(n):
    """return all the factors of n"""
    factors = set()
    for i in range(1, int(n**(0.5)) + 1):
        if not n % i:
            factors.update((i, n // i))
    return factors


def get_best_2D_segmentation(size_x, size_y, N_segments):
    """Returns (best_n_segments_x, best_n_segments_y), describing the optimal
    cartesian grid for splitting up a rectangle of size (size_x, size_y) into
    N_segments equal sized segments such as to minimise surface area between
    the segments."""
    lowest_surface_area = None
    for n_segments_x in get_factors(N_segments):
        n_segments_y = N_segments // n_segments_x
        surface_area = n_segments_x * size_y + n_segments_y * size_x
        if lowest_surface_area is None or surface_area < lowest_surface_area:
            lowest_surface_area = surface_area
            best_n_segments_x, best_n_segments_y = n_segments_x, n_segments_y
    return best_n_segments_x, best_n_segments_y


class Simulator2D(object):
    def __init__(self, x_min_global, x_max_global, y_min_global, y_max_global,
                 n_elements_x_global, n_elements_y_global, Nx, Ny, n_components,
                 output_file=None, resume=False, natural_units=False):
        """A class for simulating a the nonlinear Schrodinger equation in two
        spatial dimensions with the finite element discrete variable
        representation, on multiple cores if using MPI"""
        if (n_elements_x_global % 2):
            raise ValueError("Odd-even split step method requires even n_elements_x_global")
        if (n_elements_y_global % 2):
            raise ValueError("Odd-even split step method requires even n_elements_y_global")
        self.x_min_global = x_min_global
        self.x_max_global = x_max_global
        self.y_min_global = y_min_global
        self.y_max_global = y_max_global
        self.n_elements_x_global = n_elements_x_global
        self.n_elements_y_global = n_elements_y_global
        self.Nx = Nx
        self.Ny = Ny
        self.n_components = n_components
        self.output_file = output_file
        self.resume = resume
        self.element_width_x = (self.x_max_global - self.x_min_global)/self.n_elements_x_global
        self.element_width_y = (self.y_max_global - self.y_min_global)/self.n_elements_y_global

        self._setup_MPI_grid()

        self.elements = FiniteElements2D(self.n_elements_x, self.n_elements_y, Nx, Ny,
                                         n_components, self.x_min, self.x_max, self.y_min, self.y_max)

        self.shape = self.elements.shape
        self.global_shape = (self.n_elements_x_global, self.n_elements_y_global, self.Nx, self.Ny, self.n_components, 1)

        # Derivative operators, shapes (Nx, 1, 1, 1, Nx) and (Ny, 1, 1, Ny):
        self.gradx, self.grady = self.elements.derivative_operators()
        self.grad2x, self.grad2y = self.elements.second_derivative_operators()

        # Density operator. Is diagonal and so is represented as an (Nx, Ny, 1, 1)
        # array containing its diagonals:
        self.density_operator = self.elements.density_operator()

        # The x spatial points of the DVR basis functions, an (n_elements_x, 1, Nx, 1, 1, 1) array:
        self.x = self.elements.points_x
        # The y spatial points of the DVR basis functions, an (n_elements_y, 1, Ny, 1, 1) array:
        self.y = self.elements.points_y

        if natural_units:
            self.hbar = 1
        else:
            self.hbar = 1.054571726e-34

        if self.output_file is not None and not (os.path.exists(self.output_file) and self.resume):
            with h5py.File(self.output_file, 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:
                f.attrs['x_min_global'] = x_min_global
                f.attrs['x_max_global'] = x_max_global
                f.attrs['y_min_global'] = y_min_global
                f.attrs['y_max_global'] = y_max_global
                f.attrs['element_width_x'] = self.element_width_x
                f.attrs['element_width_y'] = self.element_width_y
                f.attrs['n_elements_x_global'] = self.n_elements_x_global
                f.attrs['n_elements_y_global'] = self.n_elements_y_global
                f.attrs['Nx'] = Nx
                f.attrs['Ny'] = Ny
                f.attrs['global_shape'] = (self.n_elements_x_global, self.n_elements_y_global, Nx, Ny)
                geometry_dtype = [('rank', int),
                                  ('processor_name', 'a256'),
                                  ('x_cart_coord', int),
                                  ('y_cart_coord', int),
                                  ('first_element_x', int),
                                  ('first_element_y', int),
                                  ('n_elements_x', int),
                                  ('n_elements_y', int)]
                MPI_geometry_dset = f.create_dataset('MPI_geometry', shape=(self.MPI_size,), dtype=geometry_dtype)
                MPI_geometry_dset.attrs['MPI_size'] = self.MPI_size
                data = (self.MPI_rank, self.processor_name,self.MPI_x_coord, self.MPI_y_coord,
                        self.global_first_x_element, self.global_first_y_element, self.n_elements_x, self.n_elements_y)
                MPI_geometry_dset[self.MPI_rank] = data
                f.create_dataset('x', data=self.x)
                f.create_dataset('y', data=self.y)
                f.create_group('output')

        # A dictionary for keeping track of what row we're up to in file
        # output in each group:
        self.output_row = {}

        # Slices for convenient indexing:
        EDGE_POINTS_X = np.s_[::self.Nx-1]
        EDGE_POINTS_Y = np.s_[::self.Nx-1]
        BOUNDARY_ELEMENTS_X = np.s_[::self.n_elements_x-1]
        BOUNDARY_ELEMENTS_Y = np.s_[::self.n_elements_x-1]

        # These are for indexing all edges of all boundary elements. The below
        # four sets of slices used in succession cover the edges of boundary
        # elements exactly once:
        self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_X = (BOUNDARY_ELEMENTS_X, ALL, EDGE_POINTS_X, ALL)
        self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_X = (INTERIOR, BOUNDARY_ELEMENTS_Y, EDGE_POINTS_X, ALL)
        self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_Y = (BOUNDARY_ELEMENTS_X, ALL, INTERIOR, EDGE_POINTS_Y)
        self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_Y = (INTERIOR, BOUNDARY_ELEMENTS_Y, INTERIOR, EDGE_POINTS_Y)

        # These are for indexing all edges of non-boundary elements. Used in
        # succession they cover the edges of these elements exactly once:
        self.INTERIOR_ELEMENTS_EDGE_POINTS_X = (INTERIOR, INTERIOR, EDGE_POINTS_X, ALL)
        self.INTERIOR_ELEMENTS_EDGE_POINTS_Y = (INTERIOR, INTERIOR, INTERIOR, EDGE_POINTS_Y)

        # These are for indexing all points of boundary elements. Used in
        # succession they cover the these elements exactly once:
        self.BOUNDARY_ELEMENTS_X_ALL_POINTS = (BOUNDARY_ELEMENTS_X, ALL, ALL, ALL)
        self.BOUNDARY_ELEMENTS_Y_ALL_POINTS = (INTERIOR, BOUNDARY_ELEMENTS_Y, ALL, ALL)

    def _setup_MPI_grid(self):
        """Split space up according to the number of MPI tasks. Set instance
        attributes for spatial extent and number of elements in this MPI task,
        and create buffers and persistent communication requests for sending
        data to adjacent processes"""

        self.MPI_size = MPI.COMM_WORLD.Get_size()
        self.MPI_size_x, self.MPI_size_y = get_best_2D_segmentation(
                                               self.n_elements_x_global, self.n_elements_y_global, self.MPI_size)
        self.MPI_comm = MPI.COMM_WORLD.Create_cart([self.MPI_size_x, self.MPI_size_y], periods=[True, True], reorder=True)
        self.MPI_rank = self.MPI_comm.Get_rank()
        self.MPI_x_coord, self.MPI_y_coord = self.MPI_comm.Get_coords(self.MPI_rank)
        self.MPI_rank_left = self.MPI_comm.Get_cart_rank((self.MPI_x_coord - 1, self.MPI_y_coord))
        self.MPI_rank_right = self.MPI_comm.Get_cart_rank((self.MPI_x_coord + 1, self.MPI_y_coord))
        self.MPI_rank_down = self.MPI_comm.Get_cart_rank((self.MPI_x_coord, self.MPI_y_coord - 1))
        self.MPI_rank_up = self.MPI_comm.Get_cart_rank((self.MPI_x_coord, self.MPI_y_coord + 1))
        self.processor_name = MPI.Get_processor_name()

        # We need an even number of elements in each direction per process. So let's share them out.
        x_elements_per_process, remaining_x_elements = (int(2*n)
                                                        for n in divmod(self.n_elements_x_global / 2, self.MPI_size_x))
        self.n_elements_x = x_elements_per_process
        if self.MPI_x_coord < remaining_x_elements/2:
            # Give the remaining to the lowest ranked processes:
            self.n_elements_x += 2

        y_elements_per_process, remaining_y_elements = (int(2*n)
                                                        for n in divmod(self.n_elements_y_global / 2, self.MPI_size_y))
        self.n_elements_y = y_elements_per_process
        if self.MPI_y_coord < remaining_y_elements/2:
            # Give the remaining to the lowest ranked processes:
            self.n_elements_y += 2

        # Where in the global array of elements are we?
        self.global_first_x_element = x_elements_per_process * self.MPI_x_coord
        # Include the extra elements some tasks have:
        if self.MPI_x_coord < remaining_x_elements/2:
            self.global_first_x_element += 2*self.MPI_x_coord
        else:
            self.global_first_x_element += remaining_x_elements

        self.global_first_y_element = y_elements_per_process * self.MPI_y_coord
        # Include the extra elements some tasks have:
        if self.MPI_y_coord < remaining_y_elements/2:
            self.global_first_y_element += 2*self.MPI_y_coord
        else:
            self.global_first_y_element += remaining_y_elements

        self.x_min = self.x_min_global + self.element_width_x * self.global_first_x_element
        self.x_max = self.x_min + self.element_width_x * self.n_elements_x

        self.y_min = self.y_min_global + self.element_width_y * self.global_first_y_element
        self.y_max = self.y_min + self.element_width_y * self.n_elements_y


        # The data we want to send to adjacent processes isn't in contiguous
        # memory, so we need to copy it into and out of temporary buffers:

        # Buffers for operating on psi with operators that are non-diagonal in
        # the spatial basis, requiring summing contributions from adjacent
        # elements:
        self.MPI_left_kinetic_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_left_kinetic_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_kinetic_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_kinetic_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_top_kinetic_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_top_kinetic_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_bottom_kinetic_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_bottom_kinetic_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)

        # Buffers for sending values of psi to adjacent processes. Values are
        # supposed to be identical on edges of adjacent elements, but due to
        # rounding error they may not stay perfectly identical. This can be a
        # problem, so we send the values across from one to the other once a
        # timestep to keep them agreeing.
        self.MPI_left_values_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_values_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_bottom_values_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_top_values_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)

        # We need to tag our data to have a way other than rank to distinguish
        # between multiple messages the two tasks might be sending each other
        # at the same time:
        TAG_LEFT_TO_RIGHT_NONDIAGS = 0
        TAG_RIGHT_TO_LEFT_NONDIAGS = 1
        TAG_DOWN_TO_UP_NONDIAGS = 2
        TAG_UP_TO_DOWN_NONDIAGS = 3
        TAG_RIGHT_TO_LEFT_VALUES = 4
        TAG_UP_TO_DOWN_VALUES = 5

        # Create persistent requests for the data transfers we will regularly be doing:
        self.MPI_send_kinetic_left = self.MPI_comm.Send_init(self.MPI_left_kinetic_send_buffer,
                                                             self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT_NONDIAGS)
        self.MPI_send_kinetic_right = self.MPI_comm.Send_init(self.MPI_right_kinetic_send_buffer,
                                                              self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT_NONDIAGS)
        self.MPI_receive_kinetic_left = self.MPI_comm.Recv_init(self.MPI_left_kinetic_receive_buffer,
                                                                self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT_NONDIAGS)
        self.MPI_receive_kinetic_right = self.MPI_comm.Recv_init(self.MPI_right_kinetic_receive_buffer,
                                                                 self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT_NONDIAGS)
        self.MPI_send_kinetic_bottom = self.MPI_comm.Send_init(self.MPI_bottom_kinetic_send_buffer,
                                                               self.MPI_rank_down, tag=TAG_UP_TO_DOWN_NONDIAGS)
        self.MPI_send_kinetic_top = self.MPI_comm.Send_init(self.MPI_top_kinetic_send_buffer,
                                                            self.MPI_rank_up, tag=TAG_DOWN_TO_UP_NONDIAGS)
        self.MPI_receive_kinetic_bottom = self.MPI_comm.Recv_init(self.MPI_bottom_kinetic_receive_buffer,
                                                                  self.MPI_rank_down, tag=TAG_DOWN_TO_UP_NONDIAGS)
        self.MPI_receive_kinetic_top = self.MPI_comm.Recv_init(self.MPI_top_kinetic_receive_buffer,
                                                               self.MPI_rank_up, tag=TAG_UP_TO_DOWN_NONDIAGS)
        self.MPI_send_values_left = self.MPI_comm.Send_init(self.MPI_left_values_send_buffer,
                                                            self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT_VALUES)
        self.MPI_receive_values_right = self.MPI_comm.Recv_init(self.MPI_right_values_receive_buffer,
                                                                self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT_VALUES)
        self.MPI_send_values_down = self.MPI_comm.Send_init(self.MPI_bottom_values_send_buffer,
                                                            self.MPI_rank_down, tag=TAG_UP_TO_DOWN_VALUES)
        self.MPI_receive_values_up = self.MPI_comm.Recv_init(self.MPI_top_values_receive_buffer,
                                                             self.MPI_rank_up, tag=TAG_UP_TO_DOWN_VALUES)

        self.MPI_kinetic_requests = [self.MPI_send_kinetic_left, self.MPI_receive_kinetic_left,
                                     self.MPI_send_kinetic_right, self.MPI_receive_kinetic_right,
                                     self.MPI_send_kinetic_bottom, self.MPI_receive_kinetic_bottom,
                                     self.MPI_send_kinetic_top, self.MPI_receive_kinetic_top]

        self.MPI_values_requests = [self.MPI_send_values_left, self.MPI_receive_values_right,
                                    self.MPI_send_values_down, self.MPI_receive_values_up]

    def MPI_send_border_kinetic(self, Kx_psi, Ky_psi):
        """Start an asynchronous MPI send to all adjacent MPI processes,
        sending them the values of H_nondiag_psi on the borders"""
        self.MPI_left_kinetic_send_buffer[:] = Kx_psi[LEFT_BOUNDARY].reshape(self.n_elements_y * self.Ny)
        self.MPI_right_kinetic_send_buffer[:] = Kx_psi[RIGHT_BOUNDARY].reshape(self.n_elements_y * self.Ny)
        self.MPI_bottom_kinetic_send_buffer[:] = Ky_psi[BOTTOM_BOUNDARY].reshape(self.n_elements_x * self.Nx)
        self.MPI_top_kinetic_send_buffer[:] = Ky_psi[TOP_BOUNDARY].reshape(self.n_elements_x * self.Nx)
        MPI.Prequest.Startall(self.MPI_kinetic_requests)

    def MPI_receive_border_kinetic(self, Kx_psi, Ky_psi):
        """Finalise an asynchronous MPI transfer from all adjacent MPI processes,
        receiving values into H_nondiag_psi on the borders"""
        MPI.Prequest.Waitall(self.MPI_kinetic_requests)
        left_data = self.MPI_left_kinetic_receive_buffer.reshape((1, self.n_elements_y, 1, self.Ny, 1, 1))
        right_data = self.MPI_right_kinetic_receive_buffer.reshape((1, self.n_elements_y, 1, self.Ny, 1, 1))
        bottom_data = self.MPI_bottom_kinetic_receive_buffer.reshape((self.n_elements_x, 1, self.Nx, 1, 1, 1))
        top_data = self.MPI_top_kinetic_receive_buffer.reshape((self.n_elements_x, 1, self.Nx, 1, 1, 1))
        if Kx_psi.dtype == np.float:
            # MPI buffers are complex, but the data might be real (in the case
            # that SOR is being done with a real trial wavefunction), and we
            # don't want to emit a casting warning:
            left_data = left_data.real
            right_data = right_data.real
            bottom_data = bottom_data.real
            top_data = top_data.real
        Kx_psi[LEFT_BOUNDARY] += left_data
        Kx_psi[RIGHT_BOUNDARY] += right_data
        Ky_psi[BOTTOM_BOUNDARY] += bottom_data
        Ky_psi[TOP_BOUNDARY] += top_data

    def global_dot(self, vec1, vec2):
        """"Dots two vectors and sums result over MPI processes"""
        # Don't double count edges
        if vec1.shape != self.shape or vec2.shape != self.shape:
            message = ('arguments must both have shape self.shape=%s, '%str(self.shape) +
                       'but they are %s and %s'%(str(vec1.shape), str(vec2.shape)))
            raise ValueError(message)
        local_dot = np.vdot(vec1[DONT_DOUBLE_COUNT_EDGES], vec2[DONT_DOUBLE_COUNT_EDGES]).real
        local_dot = np.asarray(local_dot).reshape(1)
        result = np.zeros(1)
        self.MPI_comm.Allreduce(local_dot, result, MPI.SUM)
        return result[0]

    def compute_number(self, psi):
        return self.global_dot(psi, psi)

    def normalise(self, psi, N_2D):
        """Normalise psi to the 2D normalisation constant N_2D, which has
        units of a linear density"""
        # imposing normalisation on the wavefunction:
        ncalc = self.global_dot(psi, psi)
        psi[:] *= np.sqrt(N_2D/ncalc)

    def compute_H(self, t, psi, H, boundary_element_slices=(), internal_element_slices=(),
                  sum_at_edges=True, outarrays=None):
        """Applies the Hamiltonian H to the wavefunction psi at time t, sums the
        kinetic terms at element edges and MPI task borders, and returns the
        resulting three terms K_psi, U, andn U_nonlinear. We don't sum them
        together here because the caller may with to treat them separately in
        an energy calculation, such as dividing the nonlinear term by two before
        summing up the total energy. The latter two terms are returned without
        having been multiplied by psi.

        boundary_element_slices and internal_element slices can be provided to
        specify which points should be evaluated. If both are empty, all points
        will be evaluated."""

        # optimisation, don't create arrays if the user has provided them:
        if outarrays is None:
            Kx_psi = np.empty(psi.shape, dtype=psi.dtype)
            Ky_psi = np.empty(psi.shape, dtype=psi.dtype)
            K_psi = np.empty(psi.shape, dtype=psi.dtype)
            U_psi = np.empty(psi.shape, dtype=psi.dtype)
            U_nonlinear = np.empty(psi.shape, dtype=psi.dtype)
        else:
            Kx_psi, Ky_psi, K_psi, U_psi, U_nonlinear = outarrays

        # Compute H_psi at the boundary elements first, before firing off data
        # to other MPI tasks. Then compute H_psi on the internal elements,
        # before adding in the contributions from adjacent processes at the
        # last moment. This lets us cater to as much latency in transport as
        # possible. If the caller has provided boundary_element_slices and
        # internal_element_slices, then don't evaluate all all the points, just at the
        # ones requested. Basically pre_MPI_slices must contain any boundary
        # points that the caller requires, and for maximum efficiency should
        # contain as few as possible other points - they should be be in
        # post_MPI_slices and be evaluated after the MPI send has been done.

        if not (boundary_element_slices or internal_element_slices):
            boundary_element_slices = (self.BOUNDARY_ELEMENTS_X_ALL_POINTS, self.BOUNDARY_ELEMENTS_Y_ALL_POINTS)
            internal_element_slices = (INTERIOR_ELEMENTS,)

        # Evaluate H_psi at the boundary element slices, if any:
        for slices in boundary_element_slices:
            x_elements, y_elements, x_points, y_points = slices
            Kx, Ky, U, U_nonlinear[slices] = H(t, psi, *slices)
            Kx_psi[slices] = np.einsum('...nmci,...imcq->...nmcq', Kx,  psi[x_elements, y_elements, :, y_points])
            Ky_psi[slices] = np.einsum('...mcj,...jcq->...mcq', Ky,  psi[x_elements, y_elements, x_points, :])
            U_psi[slices] = np.einsum('...cl,...lq->...cq', U,  psi[slices])

        if boundary_element_slices and sum_at_edges:
            # Send values on the border to adjacent MPI tasks:
            self.MPI_send_border_kinetic(Kx_psi, Ky_psi)

        # Now evaluate H_psi at the internal element slices:
        for slices in internal_element_slices:
            x_elements, y_elements, x_points, y_points = slices
            Kx, Ky, U, U_nonlinear[slices] = H(t, psi, *slices)
            Kx_psi[slices] = np.einsum('...nmci,...imcq->...nmcq', Kx,  psi[x_elements, y_elements, :, y_points])
            Ky_psi[slices] = np.einsum('...mcj,...jcq->...mcq', Ky,  psi[x_elements, y_elements, x_points, :])
            U_psi[slices] = np.einsum('...cl,...lq->...cq', U,  psi[slices])

        if sum_at_edges:
            # Add contributions to Kx_psi and Ky_psi at edges shared by interior elements.
            total_at_x_edges = Kx_psi[LEFT_INTERIOR_EDGEPOINTS] + Kx_psi[RIGHT_INTERIOR_EDGEPOINTS]
            Kx_psi[LEFT_INTERIOR_EDGEPOINTS] = Kx_psi[RIGHT_INTERIOR_EDGEPOINTS] = total_at_x_edges
            total_at_y_edges = Ky_psi[BOTTOM_INTERIOR_EDGEPOINTS] + Ky_psi[TOP_INTERIOR_EDGEPOINTS]
            Ky_psi[BOTTOM_INTERIOR_EDGEPOINTS] = Ky_psi[TOP_INTERIOR_EDGEPOINTS] = total_at_y_edges

        # Add contributions to K_psi from adjacent MPI tasks, if any were computed:
        if boundary_element_slices and sum_at_edges:
            self.MPI_receive_border_kinetic(Kx_psi, Ky_psi)

        for slices in boundary_element_slices:
            K_psi[slices] = Kx_psi[slices] + Ky_psi[slices]

        for slices in internal_element_slices:
            K_psi[slices] = Kx_psi[slices] + Ky_psi[slices]

        return K_psi, U_psi, U_nonlinear

    def compute_mu(self, t, psi, H, uncertainty=False):
        """Calculate chemical potential of DVR basis wavefunction psi with
        Hamiltonian H at time t. Optionally return its uncertainty."""

        # Total Hamiltonian operator operating on psi:
        K_psi, U_psi, U_nonlinear = self.compute_H(t, psi, H)
        H_psi = K_psi + U_psi + U_nonlinear * psi

        # Total norm:
        ncalc = self.compute_number(psi)

        # Expectation value and uncertainty of Hamiltonian gives the
        # expectation value and uncertainty of the chemical potential:
        mucalc = self.global_dot(psi, H_psi)/ncalc
        if uncertainty:
            mu2calc = self.global_dot(H_psi, H_psi)/ncalc
            var_mucalc = mu2calc - mucalc**2
            if var_mucalc < 0:
                u_mucalc = 0
            else:
                u_mucalc = np.sqrt(var_mucalc)
            return mucalc, u_mucalc
        else:
            return mucalc

    def compute_energy(self, psi, V, uncertainty=False):
        """Calculate total energy of DVR basis wavefunction psi with
        potential V (same shape as psi), and optionally its uncertainty"""

        # Total kinetic energy operator operating on psi:
        K_psi = self.compute_K_psi(psi)

        density = psi.conj() * self.density_operator * psi

        # Total energy operator. Differs from total Hamiltonian in that the
        # nonlinear term is halved in order to avoid double counting the
        # interaction energy:
        E_total_psi = K_psi + (V + 0.5*self.g * density) * psi
        Ecalc = self.global_dot(psi, E_total_psi)
        if uncertainty:
            E2calc = self.global_dot(E_total_psi, E_total_psi)
            var_Ecalc = E2calc - Ecalc**2
            if var_Ecalc < 0:
                u_Ecalc = 0
            else:
                u_Ecalc = np.sqrt(var_Ecalc)
            return Ecalc, u_Ecalc
        else:
            return Ecalc

    def find_groundstate(self, psi_guess, H, mu, t=0, convergence=1e-14, relaxation_parameter=1.7,
                         output_group=None, output_interval=100, output_callback=None):
        """Find the groundstate corresponding to a particular chemical
        potential using successive over-relaxation.

        H(t, psi, x_elements, y_elements, x_points, y_points) should return
        four arrays, Kx, Ky, U, and U_nonlinear, each corresponding to
        different terms in the Hamiltonian. The first two arrays, Kx and Ky,
        should be the kinetic energy operators. These should comprise only
        linear combinations of the derivative operators provided by this
        class. They should  include any terms that contain derivatives, such
        as rotation terms with first derivatives. The second array returned
        must be the sum of terms (except the nonlinear one) that are diagonal
        in the spatial basis, i.e, no derivative operators. This is typically
        the potential and and couplings between states. The third array,
        U_nonlinear, must be the nonlinear term, and can be constructed by
        multiplying the nonlinear constant by self.density_operator.

        mu should be the desired chemical potential.

        Data will be saved to a group output_group of the output file every
        output_interval steps."""
        if not self.MPI_rank: # Only one process prints to stdout:
            print('\n==========')
            print('Beginning successive over relaxation')
            print("Target chemical potential is: " + repr(mu))
            print('==========')

        psi = np.array(psi_guess, dtype=complex)

        Kx, Ky, U, U_nonlinear = H(t, psi, *ALL_ELEMENTS_AND_POINTS)
        # Get the diagonals of the Kinetic part of the Hamiltonian, shape
        # (n_elements_x, n_elements_y, Nx, Ny, n_components, 1):
        Kx_diags = np.einsum('...nmcn->...nmc', Kx).copy()
        # Broadcast Kx_diags to be the same shape as psi:
        Kx_diags = Kx_diags.reshape(Kx_diags.shape + (1,))
        Kx_diags = np.ones((self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)) * Kx_diags
        Ky_diags = np.einsum('...mcm->...mc', Ky).copy()
        # Broadcast Ky_diags to be the same shape as psi:
        Ky_diags = Ky_diags.reshape(Ky_diags.shape + (1,))
        Ky_diags = np.ones((self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)) * Ky_diags

        # Instead of summing across edges we can just multiply by two at the
        # edges to get the full values of the diagonals there:
        Kx_diags[:, :, 0] *= 2
        Kx_diags[:, :, -1] *= 2
        Ky_diags[:, :, :, 0] *= 2
        Ky_diags[:, :, :, -1] *= 2

        K_diags = Kx_diags + Ky_diags

        # The diagonal part of U, which is just the potential at each point in
        # space, shape (n_elements_x, n_elements_y, Nx, Ny, n_components, 1):
        U_diags = np.einsum('...cc->...c', U).copy()
        # Broadcast U_diags to be the same shape as psi:
        U_diags = U_diags.reshape(U_diags.shape + (1,))
        U_diags = np.ones((self.n_elements_x, self.n_elements_y, self.Nx, self.Ny, self.n_components, 1)) * U_diags

        # Empty arrays for re-using each step:
        Kx_psi = np.zeros(psi.shape, dtype=complex)
        Ky_psi = np.zeros(psi.shape, dtype=complex)
        K_psi = np.zeros(psi.shape, dtype=complex)
        U_psi = np.zeros(psi.shape, dtype=complex)
        H_diags = np.zeros(psi.shape, dtype=complex)
        H_hollow_psi = np.zeros(psi.shape, dtype=complex)
        psi_new_GS = np.zeros(psi.shape, dtype=complex)

        # All the slices for covering the edges of the boundary elements and internal elements:
        BOUNDARY_ELEMENT_EDGES = (self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_X, self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_X,
                                   self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_Y, self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_Y)
        INTERIOR_ELEMENT_EDGES = (self.INTERIOR_ELEMENTS_EDGE_POINTS_X, self.INTERIOR_ELEMENTS_EDGE_POINTS_Y)

        # Each loop we first update the edges of each element, then we loop
        # over the internal basis points. Here we just create the list of
        # slices for selecting which points we are operating on:
        point_selections = []
        # We want something we can index psi with to select edges of elements.
        # Slices can't do that, so we have to use a boolean array for the last
        # two dimensions:
        EDGE_POINTS = np.zeros((self.Nx, self.Ny), dtype=bool)
        EDGE_POINTS[FIRST] = EDGE_POINTS[LAST] = EDGE_POINTS[:, FIRST] = EDGE_POINTS[:, LAST] = True
        EDGE_POINT_SLICES = (ALL, ALL, EDGE_POINTS)
        point_selections.append(EDGE_POINT_SLICES)
        # These indicate to do internal points:
        for j in range(1, self.Nx-1):
            for k in range(1, self.Ny-1):
                # A set of slices selecting a single non-edge point in every
                # element.
                slices = (ALL, ALL, np.s_[j:j+1], np.s_[k:k+1])
                point_selections.append(slices)

        if output_group is not None:
            with h5py.File(self.output_file, driver='mpio', comm=MPI.COMM_WORLD) as f:
                if not output_group in f['output']:
                    group = f['output'].create_group(output_group)
                    group.attrs['start_time'] = time.time()
                    group.create_dataset('psi', (0,) + self.global_shape,
                                         maxshape=(None,) + self.global_shape, dtype=psi.dtype)
                    output_log_dtype = [('step_number', int), ('mucalc', int),
                                        ('convergence', float), ('time_per_step', float)]
                    group.create_dataset('output_log', (0,), maxshape=(None,), dtype=output_log_dtype)
                else:
                    self.output_row[output_group] = group = len(f['output'][output_group]['output_log']) - 1

        def do_output():
            mucalc = self.compute_mu(t, psi, H)
            convergence_calc = abs((mucalc - mu)/mu)
            time_per_step = (time.time() - start_time)/i if i else np.nan
            message =  ('step: %d'%i +
                        '  mucalc: ' + repr(mucalc) +
                        '  convergence: %E'%convergence_calc +
                        '  time per step: %.02f'%round(1e3*time_per_step, 2) + ' ms')
            if not self.MPI_rank: # Only one process prints to stdout:
                sys.stdout.write(message + '\n')
            output_log = (i, mucalc, convergence, time_per_step)
            if output_group is not None:
                self.output(output_group, psi, output_log)
            if output_callback is not None:
                try:
                    output_callback(psi, output_log)
                except Exception:
                    traceback.print_exc()
            return convergence_calc

        i = 0
        # Output the initial state, which is the zeroth timestep.
        do_output()
        i += 1

        # Start simulating:
        start_time = time.time()
        while True:
            # We operate on all elements at once, but only some DVR basis functions at a time.
            for slices in point_selections:
                if slices is EDGE_POINT_SLICES:
                    # Evaluating H_psi on all edges of all elements. We
                    # provide self.compute_H with lists of slices so that
                    # it can evaluate on the edges of the border
                    # elements first, before doing MPI transport, so that we
                    # can cater to high latency by doing useful work during
                    # the transport:
                    self.compute_H(t, psi, H,
                                   boundary_element_slices=BOUNDARY_ELEMENT_EDGES,
                                   internal_element_slices=INTERIOR_ELEMENT_EDGES,
                                   outarrays=(Kx_psi, Ky_psi, K_psi, U_psi, U_nonlinear))
                else:
                    # Evaluate H_psi at a single DVR point in all elements, requires no MPI communication:
                    self.compute_H(t, psi, H,
                                   internal_element_slices=(slices,),
                                   sum_at_edges=False,
                                   outarrays=(Kx_psi, Ky_psi, K_psi, U_psi, U_nonlinear))

                # Diagonals of the total Hamiltonian operator at the DVR point(s):
                H_diags[slices] = K_diags[slices] + U_diags[slices] + U_nonlinear[slices]

                # Hamiltonian with diagonals subtracted off, operating on psi at the DVR point(s):
                H_hollow_psi[slices] = K_psi[slices] - K_diags[slices] * psi[slices]

                # The Gauss-Seidel prediction for the new psi at the DVR point(s):
                psi_new_GS[slices] = (mu * psi[slices] - H_hollow_psi[slices])/H_diags[slices]

                # Update psi at the DVR point(s) with overrelaxation:
                psi[slices] += relaxation_parameter * (psi_new_GS[slices] - psi[slices])

            if not i % output_interval:
                convergence_calc = do_output()
                if convergence_calc < convergence:
                    if not self.MPI_rank: # Only one process prints to stdout
                        print('Convergence reached')
                    break
            i += 1

        # Output the final state if we haven't already this timestep:
        if i % output_interval:
            do_output()
        if output_group is not None:
            with h5py.File(self.output_file, driver='mpio', comm=MPI.COMM_WORLD) as f:
                group = f['output'][output_group]
                group.attrs['completion_time'] = time.time()
                group.attrs['run_time'] = group.attrs['completion_time'] - group.attrs['start_time']
        # Return complex array ready for time evolution:
        psi = np.array(psi, dtype=complex)
        return psi


    def evolve(self, psi, V, t_initial=0, t_final=np.inf, imaginary_time=False, timestep_factor=1,
               output_group=None, output_interval=100, output_callback=None, rk4=False):
        dx_min = np.diff(self.x[0, 0, :, 0]).min()
        dy_min = np.diff(self.y[0, 0, 0, :]).min()
        dt = timestep_factor * min(dx_min, dy_min)**2 * self.m / (2 * hbar)

        if not self.MPI_rank: # Only one process prints to stdout:
            print('\n==========')
            if imaginary_time:
                print("Beginning %s sec of imaginary time evolution"%str(t_final))
            else:
                print("Beginning %s sec of time evolution"%str(t_final))
            print('Using dt = ', dt)
            print('==========')

        n_initial = self.compute_number(psi)
        mu_initial = self.compute_mu(psi, V)

        E_initial = self.compute_energy(psi, V)

        if output_group is not None:
            with h5py.File(self.output_file, driver='mpio', comm=MPI.COMM_WORLD) as f:
                if not output_group in f['output']:
                    group = f['output'].create_group(output_group)
                    group.attrs['start_time'] = time.time()
                    group.attrs['dt'] = dt
                    group.attrs['t_final'] = t_final
                    group.attrs['imaginary_time'] = imaginary_time
                    group.create_dataset('psi', (0,) + self.global_shape,
                                         maxshape=(None,) + self.global_shape, dtype=psi.dtype)
                    output_log_dtype = [('step_number', int), ('time', float),
                                        ('number_err', float), ('energy_err', float), ('time_per_step', float)]
                    group.create_dataset('output_log', (0,), maxshape=(None,), dtype=output_log_dtype)
                else:
                    self.output_row[output_group] = group = len(f['output'][output_group]['output_log']) - 1

        def do_output():
            if imaginary_time:
                self.normalise(psi, n_initial)
            energy_err = self.compute_energy(psi, V) / E_initial - 1
            number_err = self.compute_number(psi) / n_initial - 1
            time_per_step = (time.time() - start_time) / i if i else np.nan
            outmessage = ('step: %d' % i +
                  '  t = %.01f' % round(t * 1e6,1) + ' us' +
                  '  number_err: %+.02E' % number_err +
                  '  energy_err: %+.02E' % energy_err +
                  '  time per step: %.02f' % round(1e3*time_per_step, 2) +' ms')
            if not self.MPI_rank: # Only one process prints to stdout:
                sys.stdout.write(outmessage + '\n')
            output_log = (i, t, number_err, energy_err, time_per_step)
            if output_group is not None:
                self.output(output_group, psi, output_log)
            if output_callback is not None:
                try:
                    output_callback(psi, output_log)
                except Exception:
                    traceback.print_exc()

        i = 0
        t = t_initial

        # Output the initial state, which is the zeroth timestep.
        do_output()
        i += 1

        # Start simulating:
        start_time = time.time()

        def dpsi_dt(psi):
            K_psi = self.compute_K_psi(psi)
            density[:] = (psi.conj()*self.density_operator*psi).real
            return -1j/hbar * (K_psi + (self.g * density + V - mu_initial)*psi)

        while t < t_final:
            k1 = dpsi_dt(psi)
            k2 = dpsi_dt(psi + k1*dt/2)
            k3 = dpsi_dt(psi + k2*dt/2)
            k4 = dpsi_dt(psi + k3*dt)
            psi[:] += dt/6*(k1 + 2*k2 + 2*k3 + k4)
            # Ensure endpoints are numerically identical:
            psi[:-1, :, -1, :] = psi[1:, :, 0, :]
            psi[-1, :, -1, :] = psi[0, :, 0, :]
            psi[:, :-1, :, -1] = psi[:, 1:, :, 0]
            psi[:, -1, :, -1] = psi[:, 0, :, 0] # TODO: MPI this
            t += dt
            if not i % output_interval:
                do_output()
            i += 1

        # t_final reached:
        if (i - 1) % output_interval:
            do_output()
        if output_group is not None:
            with h5py.File(self.output_file, driver='mpio', comm=MPI.COMM_WORLD) as f:
                group = f['output'][output_group]
                group.attrs['completion_time'] = time.time()
                group.attrs['run_time'] = group.attrs['completion_time'] - group.attrs['start_time']
        return psi

    def output(self, output_group, psi, output_log):
        with h5py.File(self.output_file, driver='mpio', comm=MPI.COMM_WORLD) as f:
            group = f['output'][output_group]
            psi_dataset = group['psi']
            output_log_dataset = group['output_log']

            start_x = self.global_first_x_element
            end_x = start_x + self.n_elements_x
            start_y = self.global_first_y_element
            end_y = start_y + self.n_elements_y

            output_row = self.output_row.setdefault(output_group, 0)
            psi_dataset.resize((output_row + 1,) + psi.shape)
            output_log_dataset.resize((output_row + 1,))

            psi_dataset[output_row, start_x:end_x, start_y:end_y, :, :] = psi
            output_log_dataset[output_row] = output_log

        self.output_row[output_group] += 1
