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

hbar = 1.054571726e-34
pi = np.pi


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


class BEC2DSimulator(object):
    def __init__(self, m, g, x_min_global, x_max_global, y_min_global, y_max_global, Nx, Ny,
                 n_elements_x_global, n_elements_y_global, output_file=None):
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

        # Second derivative operators, each (N, N):
        self.grad2x, self.grad2y = self.elements.second_derivative_operators()

        # Density operator. Is diagonal and so is represented as an (Nx, Ny) array
        # containing its diagonals:
        self.density_operator = self.elements.density_operator()

        # The x spatial points of the DVR basis functions, an (n_elements_x, 1, Nx, 1) array:
        self.x = self.elements.points_X
        # The y spatial points of the DVR basis functions, an (1, n_elements_y, 1, Ny) array:
        self.y = self.elements.points_Y

        # Kinetic energy operators:
        self.Kx = -hbar**2 / (2 * self.m) * self.grad2x
        self.Ky = -hbar**2 / (2 * self.m) * self.grad2y

        if self.output_file is not None and not os.path.exists(self.output_file):
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
                f.attrs['m'] = m
                f.attrs['g'] = g
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
        self.MPI_left_nondiags_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_left_nondiags_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_nondiags_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_nondiags_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_up_nondiags_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_up_nondiags_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_down_nondiags_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_down_nondiags_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)

        # Buffers for sending values of psi to adjacent processes. Values are
        # supposed to be identical on edges of adjacent elements, but due to
        # rounding error they may not stay perfectly identical. This can be a
        # problem, so we send the values across from one to the other once a
        # timestep to keep them agreeing.
        self.MPI_left_values_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_values_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_down_values_send_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)
        self.MPI_up_values_receive_buffer = np.zeros((self.Nx * self.n_elements_x), dtype=complex)

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
        self.MPI_send_nondiags_left = self.MPI_comm.Send_init(self.MPI_left_nondiags_send_buffer,
                                                              self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT_NONDIAGS)
        self.MPI_send_nondiags_right = self.MPI_comm.Send_init(self.MPI_right_nondiags_send_buffer,
                                                               self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT_NONDIAGS)
        self.MPI_receive_nondiags_left = self.MPI_comm.Recv_init(self.MPI_left_nondiags_receive_buffer,
                                                                 self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT_NONDIAGS)
        self.MPI_receive_nondiags_right = self.MPI_comm.Recv_init(self.MPI_right_nondiags_receive_buffer,
                                                                  self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT_NONDIAGS)
        self.MPI_send_nondiags_down = self.MPI_comm.Send_init(self.MPI_down_nondiags_send_buffer,
                                                              self.MPI_rank_down, tag=TAG_UP_TO_DOWN_NONDIAGS)
        self.MPI_send_nondiags_up = self.MPI_comm.Send_init(self.MPI_up_nondiags_send_buffer,
                                                            self.MPI_rank_up, tag=TAG_DOWN_TO_UP_NONDIAGS)
        self.MPI_receive_nondiags_down = self.MPI_comm.Recv_init(self.MPI_down_nondiags_receive_buffer,
                                                                 self.MPI_rank_down, tag=TAG_DOWN_TO_UP_NONDIAGS)
        self.MPI_receive_nondiags_up = self.MPI_comm.Recv_init(self.MPI_up_nondiags_receive_buffer,
                                                               self.MPI_rank_up, tag=TAG_UP_TO_DOWN_NONDIAGS)
        self.MPI_send_values_left = self.MPI_comm.Send_init(self.MPI_left_values_send_buffer,
                                                            self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT_VALUES)
        self.MPI_receive_values_right = self.MPI_comm.Recv_init(self.MPI_right_values_receive_buffer,
                                                                self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT_VALUES)
        self.MPI_send_values_down = self.MPI_comm.Send_init(self.MPI_down_values_send_buffer,
                                                            self.MPI_rank_down, tag=TAG_UP_TO_DOWN_VALUES)
        self.MPI_receive_values_up = self.MPI_comm.Recv_init(self.MPI_up_values_receive_buffer,
                                                             self.MPI_rank_up, tag=TAG_UP_TO_DOWN_VALUES)

        self.MPI_nondiags_requests = [self.MPI_send_nondiags_left, self.MPI_receive_nondiags_left,
                                     self.MPI_send_nondiags_right, self.MPI_receive_nondiags_right,
                                     self.MPI_send_nondiags_down, self.MPI_receive_nondiags_down,
                                     self.MPI_send_nondiags_up, self.MPI_receive_nondiags_up]

        self.MPI_values_requests = [self.MPI_send_values_left, self.MPI_receive_values_right,
                                    self.MPI_send_values_down, self.MPI_receive_values_up]

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

    def compute_K_psi(self, psi):
        """Operate on psi with the total kinetic energy operator and return
        the result."""

        # Compute Kx_psi and Ky_psi at the border elements first, before
        # firing off data to other MPI tasks. Then compute Kx_psi and Ky_psi
        # on the internal elements, before adding in the contributions from
        # adjacent processes at the last moment. This lets us cater to as much
        # latency in transport as possible.

        Kx_psi = np.empty(psi.shape, dtype=psi.dtype)
        Ky_psi = np.empty(psi.shape, dtype=psi.dtype)

        # Evaluating Kx_psi and Ky_psi on the border elements first:
        Kx_psi[(0, -1), :, :, :] = np.einsum('ij,xyjl->xyil', self.Kx,  psi[(0, -1), :, :, :])
        Ky_psi[:, (0, -1), :, :] = np.einsum('kl,xyjl->xyjk', self.Ky,  psi[:, (0, -1), :, :])

        # Send values on the border to adjacent MPI tasks:
        self.MPI_left_send_buffer[:] = Kx_psi[0, :, 0, :].reshape(self.n_elements_y * self.Ny)
        self.MPI_right_send_buffer[:] = Kx_psi[-1, :, -1, :].reshape(self.n_elements_y * self.Ny)
        self.MPI_down_send_buffer[:] = Ky_psi[:, 0, :, 0].reshape(self.n_elements_x * self.Nx)
        self.MPI_up_send_buffer[:] = Ky_psi[:, -1, :, -1].reshape(self.n_elements_x * self.Nx)
        MPI.Prequest.Startall(self.MPI_all_requests)

        # Evaluating Kx_psi and Ky_psi at all internal elements:
        Kx_psi[1:-1, :, :, :] = np.einsum('ij,xyjl->xyil', self.Kx,  psi[1:-1, :, :, :])
        Ky_psi[:, 1:-1, :, :] = np.einsum('kl,xyjl->xyjk', self.Ky,  psi[:, 1:-1, :, :])

        # Add contributions to x and y kinetic energy from both elements adjacent to each internal edge:
        Kx_psi[1:, :, 0, :] = Kx_psi[:-1, :, -1, :] = Kx_psi[1:, :, 0, :] + Kx_psi[:-1, :, -1, :]
        Ky_psi[:, 1:, :, 0] = Ky_psi[:, :-1, :, -1] = Ky_psi[:, 1:, :, 0] + Ky_psi[:, :-1, :, -1]

        # Add contributions to x and y kinetic energy from adjacent MPI tasks:
        MPI.Prequest.Waitall(self.MPI_all_requests)
        left_data = self.MPI_left_receive_buffer.reshape((self.n_elements_y, self.Ny))
        right_data = self.MPI_right_receive_buffer.reshape((self.n_elements_y, self.Ny))
        down_data = self.MPI_down_receive_buffer.reshape((self.n_elements_x, self.Nx))
        up_data = self.MPI_up_receive_buffer.reshape((self.n_elements_x, self.Nx))

        if psi.dtype == np.float:
            # MPI buffers are complex, but the data might be real (in the case
            # that SOR is being done with a real trial wavefunction), and we
            # don't want to emit a casting warning:
            left_data = left_data.real
            right_data = right_data.real
            down_data = down_data.real
            up_data = up_data.real
        Kx_psi[0, :, 0, :] += left_data
        Kx_psi[-1, :, -1, :] += right_data
        Ky_psi[:, 0, :, 0] += down_data
        Ky_psi[:, -1, :, -1] += up_data

        return Kx_psi + Ky_psi

    def compute_mu(self, psi, V, uncertainty=False):
        """Calculate chemical potential of DVR basis wavefunction psi with
        potential V (same shape as psi), and optionally its uncertainty."""

        # Total kinetic energy operator operating on psi:
        K_psi = self.compute_K_psi(psi)

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

    def find_groundstate(self, psi_guess, V, mu, convergence=1e-14, relaxation_parameter=1.7,
                         output_group=None, output_interval=100, output_callback=None):
        """Find the groundstate of the given spatial potential V with chemical
        potential mu. Data will be saved to a group output_group of the output
        file every output_interval steps."""
        if not self.MPI_rank: # Only one process prints to stdout:
            print('\n==========')
            print('Beginning successive over relaxation')
            print("Target chemical potential is: " + repr(mu))
            print('==========')

        # We use a real array, as the arithmetic is twice as fast. Groundstate
        # will be real anyway. This precludes attempting to find the
        # groundstate subject to some phase constraint. This method doesn't
        # work very well for that anyway, so use imaginary time evolution if
        # that's what you want to do.
        psi = np.abs(psi_guess)

        # Kinetic energy operators:
        Kx = self.Kx
        Ky = self.Ky

        # Diagonals of the total kinetic energy operator, shape (Nx, Ny):
        Kx_diags = Kx.diagonal().copy()
        Kx_diags[0] *= 2
        Kx_diags[-1] *= 2
        Ky_diags = Ky.diagonal().copy()
        Ky_diags[0] *= 2
        Ky_diags[-1] *= 2
        K_diags = Kx_diags[:, np.newaxis] + Ky_diags[np.newaxis, :]

        # Arrays for storing the x and y kinetic energy terms, which are
        # needed by neighboring points at edges:
        Kx_psi = np.einsum('ij,xyjl->xyil', Kx,  psi)
        Ky_psi = np.einsum('kl,xyjl->xyjk', Ky,  psi)

        j_indices = np.array(range(self.Nx))[:, np.newaxis]*np.ones(self.Ny)[np.newaxis, :]
        k_indices = np.array(range(self.Ny))[np.newaxis, :]*np.ones(self.Ny)[:, np.newaxis]
        edges = (j_indices == 0) | (k_indices == 0) | (j_indices == self.Nx - 1) | (k_indices == self.Ny - 1)

        # Each loop we first update the edges of each element, then we loop
        # over the internal basis points. Here we just create the list of
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
            for points, j, k, in points_and_indices:
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
                    self.MPI_left_send_buffer[:] = Kx_psi[0, :, 0, :].reshape(self.n_elements_y * self.Ny)
                    self.MPI_right_send_buffer[:] = Kx_psi[-1, :, -1, :].reshape(self.n_elements_y * self.Ny)
                    self.MPI_down_send_buffer[:] = Ky_psi[:, 0, :, 0].reshape(self.n_elements_x * self.Nx)
                    self.MPI_up_send_buffer[:] = Ky_psi[:, -1, :, -1].reshape(self.n_elements_x * self.Nx)
                    MPI.Prequest.Startall(self.MPI_all_requests)

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

                    # Add contributions to x and y kinetic energy from adjacent MPI tasks:
                    MPI.Prequest.Waitall(self.MPI_all_requests)
                    left_data = self.MPI_left_receive_buffer.reshape((self.n_elements_y, self.Ny))
                    right_data = self.MPI_right_receive_buffer.reshape((self.n_elements_y, self.Ny))
                    down_data = self.MPI_down_receive_buffer.reshape((self.n_elements_x, self.Nx))
                    up_data = self.MPI_up_receive_buffer.reshape((self.n_elements_x, self.Nx))
                    if psi.dtype is not float:
                        # MPI buffers are complex, but the data might be real (in the case
                        # that SOR is being done with a real trial wavefunction), and we
                        # don't want to emit a casting warning:
                        left_data = left_data.real
                        right_data = right_data.real
                        down_data = down_data.real
                        up_data = up_data.real
                    Kx_psi[0, :, 0, :] += left_data
                    Kx_psi[-1, :, -1, :] += right_data
                    Ky_psi[:, 0, :, 0] += down_data
                    Ky_psi[:, -1, :, -1] += up_data
                else:
                    # x and y kinetic energy operator operating on psi at one internal point:
                    Kx_psi[:, :, j, k] = np.einsum('j,xyj->xy', Kx[j, :],  psi[:, :, :, k])
                    Ky_psi[:, :, j, k] = np.einsum('l,xyl->xy', Ky[k, :],  psi[:, :, j, :])

                # Total kinetic energy operator operating on psi at the DVR point(s):
                K_psi = Kx_psi[:, :, points] + Ky_psi[:, :, points]

                # Density at the DVR point(s):
                density = psi[:, :, points].conj() * self.density_operator[points] * psi[:, :, points]

                # Diagonals of the total Hamiltonian operator at the DVR point(s):
                H_diags = K_diags[points] + V[:, :, points] + self.g * density

                # Hamiltonian with diagonals subtracted off, operating on psi at the DVR point(s):
                H_hollow_psi = K_psi - K_diags[points] * psi[:, :, points]

                # The Gauss-Seidel prediction for the new psi at the DVR point(s):
                psi_new_GS = (mu * psi[:, :, points] - H_hollow_psi)/H_diags

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
        dt = timestep_factor * min(dx_min, dy_min)**2 * self.m / (2 * pi * hbar)

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
