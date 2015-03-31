from __future__ import division, print_function
import os
import sys
import time
import traceback

import numpy as np
from scipy.linalg import expm
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

hbar = 1.054571726e-34
pi = np.pi

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
TOP_BOUNDARY = (ALL, LAST, ALL, LAST) # The points on the top edge of the top boundary element
INTERIOR_ELEMENTS = (INTERIOR, INTERIOR, ALL, ALL) # All points in all elements that are not boundary elements
INTERIOR_POINTS = (ALL, ALL, INTERIOR, INTERIOR) # Points that are not edgepoints in all elements
LEFT_INTERIOR_EDGEPOINTS = (ALL_BUT_FIRST, ALL, FIRST, ALL) # All points on the left edge of a non-border element
RIGHT_INTERIOR_EDGEPOINTS = (ALL_BUT_LAST, ALL, LAST, ALL) # All points on the right edge of a non-border element
BOTTOM_INTERIOR_EDGEPOINTS = (ALL, ALL_BUT_FIRST, ALL, FIRST) # All points on the bottom edge of a non-border element
TOP_INTERIOR_EDGEPOINTS = (ALL, ALL_BUT_LAST, ALL, LAST) # All points on the top edge of a non-border element

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
    def __init__(self, m, g, x_min_global, x_max_global, y_min_global, y_max_global, Nx, Ny,
                 n_elements_x_global, n_elements_y_global, output_file=None, resume=True, natural_units=False):
        """A class for simulating the Gross-Pitaevskii equation in 2D with the
        finite element discrete variable representation, on multiple cores if
        using MPI"""
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
        self.output_file = output_file
        self.Nx = Nx
        self.Ny = Ny
        self.m = m
        self.g = g
        self.element_width_x = (self.x_max_global - self.x_min_global)/self.n_elements_x_global
        self.element_width_y = (self.y_max_global - self.y_min_global)/self.n_elements_y_global

        self._setup_MPI_grid()

        self.elements = FiniteElements2D(Nx, Ny, self.n_elements_x, self.n_elements_y,
                                         self.x_min, self.x_max, self.y_min, self.y_max)

        self.shape = self.elements.shape
        self.global_shape = (self.n_elements_x_global, self.n_elements_y_global, self.Nx, self.Ny)

        # Derivative operators, each (N, N):
        self.gradx, self.grady = self.elements.derivative_operators()
        self.grad2x, self.grad2y = self.elements.second_derivative_operators()

        # Density operator. Is diagonal and so is represented as an (Nx, Ny)
        # array containing its diagonals:
        self.density_operator = self.elements.density_operator()

        # The x spatial points of the DVR basis functions, an (n_elements_x, 1, Nx, 1) array:
        self.x = self.elements.points_X
        # The y spatial points of the DVR basis functions, an (1, n_elements_y, 1, Ny) array:
        self.y = self.elements.points_Y

        if natural_units:
            self.hbar = 1
        else:
            self.hbar = 1.054571726e-34

        # Kinetic energy operators:
        self.Kx = -hbar**2 / (2 * self.m) * self.grad2x
        self.Ky = -hbar**2 / (2 * self.m) * self.grad2y

        if self.output_file is not None:
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
        # elements exactly once without doubling up on corner points:
        self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_X = (BOUNDARY_ELEMENTS_X, ALL, EDGE_POINTS_X, ALL)
        self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_X = (INTERIOR, BOUNDARY_ELEMENTS_Y, EDGE_POINTS_X, ALL)
        self.BOUNDARY_ELEMENTS_X_EDGE_POINTS_Y = (BOUNDARY_ELEMENTS_X, ALL, INTERIOR, EDGE_POINTS_Y)
        self.BOUNDARY_ELEMENTS_Y_EDGE_POINTS_Y = (INTERIOR, BOUNDARY_ELEMENTS_Y, INTERIOR, EDGE_POINTS_Y)

        # These are for indexing all edges of non-boundary elements. Used in
        # succession they cover the edges of these elements exactly once
        # without doubling up on corner points.
        self.INTERIOR_ELEMENTS_EDGE_POINTS_X = (INTERIOR, INTERIOR, EDGE_POINTS_X, ALL)
        self.INTERIOR_ELEMENTS_EDGE_POINTS_Y = (INTERIOR, INTERIOR, INTERIOR, EDGE_POINTS_Y)

        # These are for indexing all points of border elements. Used in
        # succession they cover the edges of these elements exactly once
        # without doubling up on corner elements.
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
        self.MPI_left_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_left_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_up_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_up_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_down_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_down_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)

        # We need to tag our data in case there are exactly two MPI tasks in a
        # direction, in which case we need a way other than rank to
        # distinguish between the two messages the two tasks might be sending
        # each other at one time:
        TAG_LEFT_TO_RIGHT = 0
        TAG_RIGHT_TO_LEFT = 1
        TAG_DOWN_TO_UP = 2
        TAG_UP_TO_DOWN = 3

        # Create persistent requests for the data transfers we will regularly be doing:
        self.MPI_send_left = self.MPI_comm.Send_init(self.MPI_left_send_buffer,
                                                     self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_send_right = self.MPI_comm.Send_init(self.MPI_right_send_buffer,
                                                      self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_left = self.MPI_comm.Recv_init(self.MPI_left_receive_buffer,
                                                        self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_right = self.MPI_comm.Recv_init(self.MPI_right_receive_buffer,
                                                         self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_send_down = self.MPI_comm.Send_init(self.MPI_down_send_buffer,
                                                     self.MPI_rank_down, tag=TAG_UP_TO_DOWN)
        self.MPI_send_up = self.MPI_comm.Send_init(self.MPI_up_send_buffer,
                                                      self.MPI_rank_up, tag=TAG_DOWN_TO_UP)
        self.MPI_receive_down = self.MPI_comm.Recv_init(self.MPI_down_receive_buffer,
                                                        self.MPI_rank_down, tag=TAG_DOWN_TO_UP)
        self.MPI_receive_up = self.MPI_comm.Recv_init(self.MPI_up_receive_buffer,
                                                         self.MPI_rank_up, tag=TAG_UP_TO_DOWN)

        self.MPI_all_requests = [self.MPI_send_left, self.MPI_receive_left,
                                 self.MPI_send_right, self.MPI_receive_right,
                                 self.MPI_send_down, self.MPI_receive_down,
                                 self.MPI_send_up, self.MPI_receive_up]

        self.MPI_left_to_right_requests = [self.MPI_send_right, self.MPI_receive_left]
        self.MPI_right_to_left_requests = [self.MPI_send_left, self.MPI_receive_right]
        self.MPI_down_to_up_requests = [self.MPI_send_up, self.MPI_receive_down]
        self.MPI_up_to_down_requests = [self.MPI_send_down, self.MPI_receive_up]

    def MPI_send_border_kinetic(self, Kx_psi, Ky_psi):
        """Start an asynchronous MPI send to all adjacent MPI processes,
        sending them the values of H_nondiag_psi on the borders"""
        self.MPI_left_send_buffer[:] = Kx_psi[LEFT_BOUNDARY].reshape(self.n_elements_y * self.Ny)
        self.MPI_right_send_buffer[:] = Kx_psi[RIGHT_BOUNDARY].reshape(self.n_elements_y * self.Ny)
        self.MPI_down_send_buffer[:] = Ky_psi[BOTTOM_BOUNDARY].reshape(self.n_elements_x * self.Nx)
        self.MPI_up_send_buffer[:] = Ky_psi[TOP_BOUNDARY].reshape(self.n_elements_x * self.Nx)
        MPI.Prequest.Startall(self.MPI_all_requests)

    def MPI_receive_border_kinetic(self, Kx_psi, Ky_psi):
        """Finalise an asynchronous MPI transfer from all adjacent MPI processes,
        receiving values into H_nondiag_psi on the borders"""
        MPI.Prequest.Waitall(self.MPI_all_requests)
        left_data = self.MPI_left_receive_buffer.reshape((1, self.n_elements_y, 1, self.Ny))
        right_data = self.MPI_right_receive_buffer.reshape((1, self.n_elements_y, 1, self.Ny))
        bottom_data = self.MPI_down_receive_buffer.reshape((self.n_elements_x, 1, self.Nx, 1))
        top_data = self.MPI_up_receive_buffer.reshape((self.n_elements_x, 1, self.Nx, 1))
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
        local_dot = np.vdot(vec1[:, :, :-1, :-1], vec2[:, :, :-1, :-1]).real
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

    def compute_H(self, psi, boundary_element_slices=None, internal_element_slices=None, outarrays=None):
        """Operate on psi with the total kinetic energy operator and return
        the result."""
        """Applies the Hamiltonian to psi with H_psi, sums the nondiagonal
        contributions at element edges and MPI task borders, and returns the
        resulting three terms H_nondiag_psi, H_diag_psi, nonlinear_psi. We
        don't sum them together here because the caller may with to treat them
        separately in an energy calculation, such as dividing the nonlinear
        term by two before summing up the total energy.

        boundary_element_slices and internal_element slices can be provided to
        specify which points should be evaluated. If both are None, all points
        will be evaluated. Behaviour unspecified if only one is None."""

        # optimisation, don't create arrays if the user has provided them:
        if outarrays is None:
            Kx_psi = np.empty(psi.shape, dtype=psi.dtype)
            Ky_psi = np.empty(psi.shape, dtype=psi.dtype)
            K_psi = np.empty(psi.shape, dtype=psi.dtype)
            U = np.empty(psi.shape, dtype=psi.dtype)
            U_nonlinear = np.empty(psi.shape, dtype=psi.dtype)
        else:
            Kx_psi, Ky_psi, K_psi, U, U_nonlinear = outarrays

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

        # Evaluating Kx_psi and Ky_psi on the border elements first:
        Kx_psi[(0, -1), :, :, :] = np.einsum('ij,xyjl->xyil', self.Kx,  psi[(0, -1), :, :, :])
        Ky_psi[:, (0, -1), :, :] = np.einsum('kl,xyjl->xyjk', self.Ky,  psi[:, (0, -1), :, :])

        # Send values on the border to adjacent MPI tasks:
        self.MPI_send_border_kinetic(Kx_psi, Ky_psi)

        # Evaluating Kx_psi and Ky_psi at all internal elements:
        Kx_psi[1:-1, :, :, :] = np.einsum('ij,xyjl->xyil', self.Kx,  psi[1:-1, :, :, :])
        Ky_psi[:, 1:-1, :, :] = np.einsum('kl,xyjl->xyjk', self.Ky,  psi[:, 1:-1, :, :])

        K_psi = Kx_psi + Ky_psi

        # Add contributions to x and y kinetic energy from both elements adjacent to each internal edge:
        Kx_psi[1:, :, 0, :] = Kx_psi[:-1, :, -1, :] = Kx_psi[1:, :, 0, :] + Kx_psi[:-1, :, -1, :]
        Ky_psi[:, 1:, :, 0] = Ky_psi[:, :-1, :, -1] = Ky_psi[:, 1:, :, 0] + Ky_psi[:, :-1, :, -1]

        # Add contributions to H_nondiag_psi from adjacent MPI tasks:
        self.MPI_receive_border_kinetic(Kx_psi, Ky_psi)

        return Kx_psi + Ky_psi

    def compute_mu(self, psi, V, uncertainty=False):
        """Calculate chemical potential of DVR basis wavefunction psi with
        potential V (same shape as psi), and optionally its uncertainty."""

        # Total kinetic energy operator operating on psi:
        K_psi = self.compute_H(psi)

        # Total norm:
        ncalc = self.global_dot(psi, psi)

        # Total Hamaltonian:
        density = psi.conj() * self.density_operator * psi
        H_psi = K_psi + (V + self.g * density) * psi

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

    def find_groundstate(self, psi_guess, H, V, mu, t=0, convergence=1e-14, relaxation_parameter=1.7,
                         output_group=None, output_interval=100, output_callback=None):
        """Find the groundstate corresponding to a particular chemical
        potential using successive over-relaxation.

        H(t, psi, x_elements, y_elements, x_points, y_points) should return
        three arrays, K_psi, U, and U_nonlinear, each corresponding to
        different terms in the Hamiltonian operating on psi. The first array,
        K_psi, should be the kinetic energy operator acting on psi. This
        should comprise only linear combinations of the derivative operators
        provided by this class, acting on psi. MPI communication will be
        performed on this array to include contributions from adjacent MPI
        tasks. K_psi should include any terms that contain derivatives, such
        as rotation terms with first derivatives. The second array returned
        must be the sum of terms that are diagonal in the spatial basis, i.e,
        no derivative operators, and must be linear in psi. This is typically
        the potential and and couplings between states. The third array ,
        U_nonlinear, must be the nonlinear term, and can be constructed by
        multiplying the nonlinear constant by self.density_operator. U and
        U_nonlinear should not be multiplied by psi before being returned.

        K_diags(x, y) should return the diagonals of the kinetic part of the
        Hamiltonian only, that is, the part of the Hamiltonian corresponding
        to the first array returned by H_psi (but not acting on psi).

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
        # Get the diagonals of the nondiagonal part of the Hamiltonian, shape (1, 1,
        # Nx, Ny) if spatially homogenous, otherwise first two dimensions can have
        # sizes n_elements_x or n_elements_y:
        Kx_diags = Kx.diagonal().copy()
        Ky_diags = Ky.diagonal().copy()

        # Instead of summing across edges we can just multiply by two at the
        # edges to get the full values of the diagonals there:
        Kx_diags[0] *= 2
        Kx_diags[-1] *= 2
        Ky_diags[0] *= 2
        Ky_diags[-1] *= 2
        K_diags = Kx_diags[:, np.newaxis] + Ky_diags[np.newaxis, :]


        # Empty arrays for re-using each step:
        Kx_psi = np.zeros(psi.shape, dtype=complex)
        Ky_psi = np.zeros(psi.shape, dtype=complex)
        K_psi = np.zeros(psi.shape, dtype=complex)
        U = np.zeros(psi.shape, dtype=complex)
        U_nonlinear = np.zeros(psi.shape, dtype=complex)
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
                # element .
                slices = (ALL, ALL, np.s_[j:j+1], np.s_[k:k+1])
                point_selections.append(slices)

        j_indices = np.array(range(self.Nx))[:, np.newaxis]*np.ones(self.Ny)[np.newaxis, :]
        k_indices = np.array(range(self.Ny))[np.newaxis, :]*np.ones(self.Ny)[:, np.newaxis]
        edges = (j_indices == 0) | (k_indices == 0) | (j_indices == self.Nx - 1) | (k_indices == self.Ny - 1)

        # indices we iterate over:
        points_and_indices = []
        points_and_indices.append((edges, None, None))
        for j in range(1, self.Nx-1):
            for k in range(1, self.Ny-1):
                points_and_indices.append(((j_indices == j) & (k_indices == k), j, k))

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
            mucalc = self.compute_mu(psi, V)
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
            for (points, j, k), slices in zip(points_and_indices, point_selections):
                if points is edges:

                    # First we compute the values that we will need to send to
                    # adjacent processes, and then those on the edges of
                    # internal elements. This allows us to send data with MPI
                    # as soon as possible and receive as late as possibly, so
                    # that we are still doing useful work if there is high
                    # latency.

                    # Evaluating Kx_psi on the left and right edges of the
                    # left and rightmost elements:
                    Kx_psi[:] = 0
                    Ky_psi[:] = 0
                    # Evaluating Kx_psi on the left and right edges of the
                    # left and rightmost elements:
                    Kx_psi[0, :, 0, :] = np.einsum('j,yjl->yl', Kx[0, :], psi[0, :, :, :])
                    Kx_psi[0, :, -1, :] = np.einsum('j,yjl->yl', Kx[-1, :], psi[0, :, :, :])
                    Kx_psi[-1, :, 0, :] = np.einsum('j,yjl->yl', Kx[0, :], psi[-1, :, :, :])
                    Kx_psi[-1, :, -1, :] = np.einsum('j,yjl->yl', Kx[-1, :], psi[-1, :, :, :])
                    # Evaluating Ky_psi on the upper and lower edges of the
                    # top and bottom elements:
                    Ky_psi[:, 0, :, 0] = np.einsum('l,xjl->xj', Ky[0, :], psi[:, 0, :, :])
                    Ky_psi[:, 0, :, -1] = np.einsum('l,xjl->xj', Ky[-1, :], psi[:, 0, :, :])
                    Ky_psi[:, -1, :, 0] = np.einsum('l,xjl->xj', Ky[0, :], psi[:, -1, :, :])
                    Ky_psi[:, -1, :, -1] = np.einsum('l,xjl->xj', Ky[-1, :], psi[:, -1, :, :])

                    # Send values on the border to adjacent MPI tasks:
                    self.MPI_send_border_kinetic(Kx_psi, Ky_psi)

                    # Evaluating Kx_psi at the rest of the edges we haven't
                    # done yet. Man, there's some crazy indexing going on
                    # here. The first line below is doing the left and right
                    # edges of all elements except the leftmost and rightmost
                    # elements, which we've already done. The second line is
                    # doing the top and bottom edges of all elements, but
                    # excluding the corners because they were already done by
                    # the first line.
                    Kx_psi[1:-1, :, (0, -1), :] = np.einsum('ij,xyjl->xyil', Kx[(0, -1), :],  psi[1:-1, :, :, :])
                    Kx_psi[:, :, 1:-1, (0, -1)] = np.einsum('ij,xyjl->xyil', Kx[1:-1, :],  psi[:, :, :, (0, -1)])
                    # Same for Ky_psi:
                    Ky_psi[:, 1:-1, :, (0, -1)] = np.einsum('kl,xyjl->xyjk', Ky[(0, -1), :],  psi[:, 1:-1, :, :])
                    Ky_psi[:, :, (0, -1), 1:-1] = np.einsum('kl,xyjl->xyjk', Ky[1:-1, :],  psi[:, :, (0, -1), :])

                    # Add contributions to x and y kinetic energy from both
                    # elements adjacent to each internal edge:
                    Kx_psi[1:, :, 0, :] = Kx_psi[:-1, :, -1, :] = Kx_psi[1:, :, 0, :] + Kx_psi[:-1, :, -1, :]
                    Ky_psi[:, 1:, :, 0] = Ky_psi[:, :-1, :, -1] = Ky_psi[:, 1:, :, 0] + Ky_psi[:, :-1, :, -1]

                    # Add contributions to H_nondiag_psi from adjacent MPI tasks:
                    self.MPI_receive_border_kinetic(Kx_psi, Ky_psi)

                    
                else:
                    # x and y kinetic energy operator operating on psi at one internal point:
                    Kx_psi[:, :, j, k] = np.einsum('j,xyj->xy', Kx[j, :],  psi[:, :, :, k])
                    Ky_psi[:, :, j, k] = np.einsum('l,xyl->xy', Ky[k, :],  psi[:, :, j, :])

                # Total kinetic energy operator operating on psi at the DVR point(s):
                K_psi[:, :, points] = Kx_psi[:, :, points] + Ky_psi[:, :, points]

                # Density at the DVR point(s):
                density = psi[:, :, points].conj() * self.density_operator[points] * psi[:, :, points]

                # Diagonals of the total Hamiltonian operator at the DVR point(s):
                H_diags = K_diags[points] + V[:, :, points] + self.g * density

                # Hamiltonian with diagonals subtracted off, operating on psi at the DVR point(s):
                H_hollow_psi = K_psi[:, :, points] - K_diags[points] * psi[:, :, points]

                # The Gauss-Seidel prediction for the new psi at the DVR point(s):
                psi_new_GS = (mu * psi[:, :, points] - H_hollow_psi)/H_diags

                #import IPython
                #IPython.embed()
                # Update psi at the DVR point(s) with overrelaxation:

                psi[:, :, points] += relaxation_parameter * (psi_new_GS - psi[:, :, points])

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

        # Which elements are odd numbered, and which are even?
        element_numbers_x = np.array(range(self.n_elements_x))
        element_numbers_y = np.array(range(self.n_elements_y))
        even_elements_x = (element_numbers_x % 2) == 0
        odd_elements_x = ~even_elements_x
        even_elements_y = (element_numbers_y % 2) == 0
        odd_elements_y = ~even_elements_y

        # Which elements are internal, and which are on boundaries?
        internal_elements_x = (0 < element_numbers_x) & (element_numbers_x < self.n_elements_x - 1)
        internal_elements_y = (0 < element_numbers_y) & (element_numbers_y < self.n_elements_y - 1)
        odd_internal_elements_x = odd_elements_x & internal_elements_x
        even_internal_elements_x = even_elements_x & internal_elements_x
        odd_internal_elements_y = odd_elements_y & internal_elements_y
        even_internal_elements_y = even_elements_y & internal_elements_y

        def do_odd_Kx_evolution():
            # First we evolve the right border elements before firing off data
            # to the MPI task to the right of us. Then we evolve internal
            # elements. This allows us to send data with MPI as soon as
            # possible and receive as late as possibly, so that we are still
            # doing useful work if there is high latency.

            # Evolve psi on the right border elements first:
            psi[-1, : ,:, :] = np.einsum('ij,yjl->yil', U_Kx_half_substep, psi[-1, :, :, :])

            # Send values to the right and and receive from the left adjacent
            # MPI tasks:
            self.MPI_right_send_buffer[:] = psi[-1, :, -1, :].reshape(self.n_elements_y * self.Ny)
            MPI.Prequest.Startall(self.MPI_left_to_right_requests)

            # Evolving psi at the internal odd elements:
            psi[odd_internal_elements_x, :, :, :] = np.einsum('ij,xyjl->xyil',
                                                               U_Kx_half_substep, psi[odd_internal_elements_x, :, :, :])

            # Copy values from odd -> even elements at internal edges in x
            # direction:
            psi[even_elements_x, :, -1, :] = psi[odd_elements_x, :, 0, :]
            psi[even_internal_elements_x, :, 0, :] = psi[odd_internal_elements_x, :, -1, :]

            # Copy values received from the left adjacent MPI task to the left
            # border points:
            MPI.Prequest.Waitall(self.MPI_left_to_right_requests)
            # Copy over neighbouring process's value for psi at the boundary:
            psi[0, :, 0, :] = self.MPI_left_receive_buffer.reshape((self.n_elements_y, self.Ny))

        def do_even_Kx_evolution():
            # First we evolve the left border elements before firing off data
            # to the MPI task to the left of us. Then we evolve internal
            # elements. This allows us to send data with MPI as soon as
            # possible and receive as late as possibly, so that we are still
            # doing useful work if there is high latency.

            # Evolve psi on the left border elements first:
            psi[0] = np.einsum('ij,yjl->yil', U_Kx_full_substep, psi[0])

            # Send values to the left and and receive from the right adjacent
            # MPI tasks:
            self.MPI_left_send_buffer[:] = psi[0, :, 0].reshape(self.n_elements_y * self.Ny)
            MPI.Prequest.Startall(self.MPI_right_to_left_requests)

            # Evolving psi at the internal even elements:
            psi[even_internal_elements_x] = np.einsum('ij,xyjl->xyil',
                                                      U_Kx_full_substep, psi[even_internal_elements_x])

            # Copy values from even -> odd elements at internal edges in x
            # direction:
            psi[odd_internal_elements_x, :, -1, :] = psi[even_internal_elements_x, :, 0]
            psi[odd_elements_x, :, 0] = psi[even_elements_x, :, -1]

            # Copy values received from the right adjacent MPI task to the
            # right border points:
            MPI.Prequest.Waitall(self.MPI_right_to_left_requests)
            # Copy over neighbouring process's value for psi at the boundary:
            psi[-1, :, -1] = self.MPI_right_receive_buffer.reshape((self.n_elements_y, self.Ny))

        def do_odd_Ky_evolution():
            # First we evolve the upper border elements before firing off data
            # to the MPI task above us. Then we evolve internal elements. This
            # allows us to send data with MPI as soon as possible and receive
            # as late as possibly, so that we are still doing useful work if
            # there is high latency.

            # Evolve psi on the upper border elements first:
            psi[:, -1] = np.einsum('kl,xjl->xjk', U_Ky_half_substep, psi[:, -1])

            # Send values to the upper and and receive from the lower adjacent
            # MPI tasks:
            self.MPI_up_send_buffer[:] = psi[:, -1, :, -1].reshape(self.n_elements_x * self.Nx)
            MPI.Prequest.Startall(self.MPI_down_to_up_requests)

            # Evolving psi at the internal odd elements:
            psi[:, odd_internal_elements_y] = np.einsum('kl,xyjl->xyjk',
                                                        U_Ky_half_substep, psi[:, odd_internal_elements_y])

            # Copy values from odd -> even elements at internal edges in y
            # direction:
            psi[:, even_elements_y, :, -1] = psi[:, odd_elements_y, :, 0]
            psi[:, even_internal_elements_y, :, 0] = psi[:, odd_internal_elements_y, :, -1]

            # Copy values received from the lower adjacent MPI task to the
            # lower border points:
            MPI.Prequest.Waitall(self.MPI_down_to_up_requests)
            psi[:, 0, :, 0] = self.MPI_down_receive_buffer.reshape((self.n_elements_x, self.Nx))

        def do_even_Ky_evolution():
            # First we evolve the lower border elements before firing off data
            # to the MPI task below us. Then we evolve internal elements. This
            # allows us to send data with MPI as soon as possible and receive
            # as late as possibly, so that we are still doing useful work if
            # there is high latency.
            psi[:, 0] = np.einsum('kl,xjl->xjk', U_Ky_full_substep, psi[:, 0])

            # Send values to the lower and and receive from the upper adjacent
            # MPI tasks:
            self.MPI_down_send_buffer[:] = psi[:, 0, :, 0].reshape(self.n_elements_x * self.Nx)
            MPI.Prequest.Startall(self.MPI_up_to_down_requests)

            # Evolving psi at the internal odd elements:
            psi[:, even_internal_elements_y] = np.einsum('kl,xyjl->xyjk',
                                                         U_Ky_full_substep, psi[:, even_internal_elements_y])

            # Copy values from even -> odd elements at internal edges in y
            # direction:
            psi[:, odd_internal_elements_y, :, -1] = psi[:, even_internal_elements_y, :, 0]
            psi[:, odd_elements_y, :, 0] = psi[:, even_elements_y, :, -1]

            # Copy values received from the upper adjacent MPI task to the
            # upper border points:
            MPI.Prequest.Waitall(self.MPI_up_to_down_requests)
            psi[:, -1, :, -1] = self.MPI_up_receive_buffer.reshape((self.n_elements_x, self.Nx))

        # Now we construct some unitary time evolution operators. We are using
        # the fourth order "real space product" split operator method, which
        # can be constructed as a series of second order evolution operators.
        # See equation 18 of "The discrete variable method for the solution of
        # the time-dependent Schrodinger equation"  by Barry I. Schneider, Lee
        # A. Collins, Journal of Non-Crystalline Solids 351 (2005). Basically
        # we need to construct unitary evolution operators for half and full
        # multiples of both p * dt and q * dt:
        p = 1. / (4 - 4 ** (1. / 3))
        q = 1 - 4 * p
        if imaginary_time:
            # The kinetic energy unitary evolution oparators for half a timestep
            # in imaginary time, shapes (Nx, Nx) and (Ny, Ny). Not diagonal, but
            # the same in each element.
            U_Kx_halfstep_p = expm(-1/hbar * self.Kx * p * dt/2)
            U_Ky_halfstep_p = expm(-1/hbar * self.Ky * p * dt/2)
            U_Kx_halfstep_q = expm(-1/hbar * self.Kx * q * dt/2)
            U_Ky_halfstep_q = expm(-1/hbar * self.Ky * q * dt/2)
            # The same as above but for a full timestep:
            U_Kx_fullstep_p = expm(-1/hbar * self.Kx * p * dt)
            U_Ky_fullstep_p = expm(-1/hbar * self.Ky * p * dt)
            U_Kx_fullstep_q = expm(-1/hbar * self.Kx * q * dt)
            U_Ky_fullstep_q = expm(-1/hbar * self.Ky * q * dt)
        else:
            # The kinetic energy unitary evolution oparators for half a timestep,
            # shapes (Nx, Nx) and (Ny, Ny). Not diagonal, but the same in each
            # element.
            U_Kx_halfstep_p = expm(-1j/hbar * self.Kx * p * dt/2)
            U_Ky_halfstep_p = expm(-1j/hbar * self.Ky * p * dt/2)
            U_Kx_halfstep_q = expm(-1j/hbar * self.Kx * q * dt/2)
            U_Ky_halfstep_q = expm(-1j/hbar * self.Ky * q * dt/2)
            # The same as above but for a full timestep:
            U_Kx_fullstep_p = expm(-1j/hbar * self.Kx * p * dt)
            U_Ky_fullstep_p = expm(-1j/hbar * self.Ky * p * dt)
            U_Kx_fullstep_q = expm(-1j/hbar * self.Kx * q * dt)
            U_Ky_fullstep_q = expm(-1j/hbar * self.Ky * q * dt)

        # The potential energy evolution operator for the first half timestep.
        # It is always the same as at the end of timesteps, so we usually just
        # re-use at the start of each loop. But this being the first loop we
        # need it now too. We don't actually need the q ones to have the right
        # values right now, but the arrays need to be defined still.
        density = (psi.conj()*self.density_operator*psi).real
        if imaginary_time:
            U_V_halfstep_p = np.exp(-1/hbar * (self.g * density + V - mu_initial) * p * dt/2)
            U_V_halfstep_q = np.exp(-1/hbar * (self.g * density + V - mu_initial) * q * dt/2)
        else:
            U_V_halfstep_p = np.exp(-1j/hbar * (self.g * density + V - mu_initial) * p * dt/2)
            U_V_halfstep_q = np.exp(-1j/hbar * (self.g * density + V - mu_initial) * q * dt/2)

        # We implement the fourth order method by repeated application of the
        # second order method with differently sized 'sub-timesteps'. Here we
        # make a list of the five sub-timestep sizes and sets of operators
        # used in each sub-step (four of the five substeps are the same):
        p_substep = (p*dt, U_V_halfstep_p, U_Kx_halfstep_p, U_Ky_halfstep_p, U_Kx_fullstep_p, U_Ky_fullstep_p)
        q_substep = (q*dt, U_V_halfstep_q, U_Kx_halfstep_q, U_Ky_halfstep_q, U_Kx_fullstep_q, U_Ky_fullstep_q)
        substeps = [p_substep, p_substep, q_substep, p_substep, p_substep]

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
            if not rk4:
                for substep_number, substep in enumerate(substeps):
                    (sub_dt, U_V_half_substep, U_Kx_half_substep,
                     U_Ky_half_substep,U_Kx_full_substep, U_Ky_full_substep) = substep

                    if substep_number in [2, 3]:
                        # If it's the q substep, or the p substep after the q
                        # substep, then we need to update our potential energy
                        # evolution operator to reflect the current density.
                        # Otherwise, it is the same as at the end of the last sub-
                        # step, and we can just re-use it:
                        if imaginary_time:
                            U_V_half_substep[:] = np.exp(-1/hbar * (self.g * density + V - mu_initial) * sub_dt / 2)
                        else:
                            U_V_half_substep[:] = np.exp(-1j/hbar * (self.g * density + V - mu_initial) * sub_dt / 2)

                    # Evolve for half a substep with potential evolution operator:
                    psi[:] = U_V_half_substep*psi

                    # Evolution with x kinetic energy evolution operator, using
                    # odd-even-odd split step method:
                    do_odd_Kx_evolution()
                    do_even_Kx_evolution()
                    do_odd_Kx_evolution()

                    # Evolution with y kinetic energy evolution operator, using
                    # odd-even-odd split step method:
                    do_odd_Ky_evolution()
                    do_even_Ky_evolution()
                    do_odd_Ky_evolution()

                    # Calculate potential energy evolution operator for half a step
                    if imaginary_time:
                        self.normalise(psi, n_initial)
                    density[:] = (psi.conj()*self.density_operator*psi).real
                    if imaginary_time:
                        U_V_half_substep[:] = np.exp(-1/hbar * (self.g * density + V - mu_initial) * sub_dt / 2)
                    else:
                        U_V_half_substep[:] = np.exp(-1j/hbar * (self.g * density + V - mu_initial) * sub_dt / 2)

                    # Evolve for half a timestep with potential evolution operator:
                    psi[:] = U_V_half_substep*psi

                    t += sub_dt
            else:
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
