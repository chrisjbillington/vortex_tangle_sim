from __future__ import division, print_function
import time

import numpy as np
from scipy.linalg import expm
from mpi4py import MPI

from FEDVR import FiniteElements2D

hbar = 1.054571726e-34
pi = np.pi

class BEC2DSimulator(object):
    def __init__(self, m, g, x_min_global, x_max_global, y_min_global, y_max_global, Nx, Ny,
                 n_elements_x_global, n_elements_y_global):
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

    def _setup_MPI_grid(self):
        """Split space up according to the number of MPI tasks. Set instance
        attributes for spatial extent and number of elements in this MPI task,
        and create buffers and persistent communication requests for sending
        data to adjacent processes"""

        self.MPI_size = MPI.COMM_WORLD.Get_size()
        self.MPI_comm = MPI.COMM_WORLD.Create_cart([self.MPI_size], periods=[True], reorder=True)
        self.MPI_rank = self.MPI_comm.Get_rank()
        self.MPI_x_coord, = self.MPI_comm.Get_coords(self.MPI_rank)
        self.MPI_rank_left = self.MPI_comm.Get_cart_rank((self.MPI_x_coord - 1,))
        self.MPI_rank_right = self.MPI_comm.Get_cart_rank((self.MPI_x_coord + 1,))

        # We need an even number of elements in the x direction per process. So let's share them out.
        elements_per_process, remaining_elements = (2*n for n in divmod(self.n_elements_x_global/2, self.MPI_size))
        self.n_elements_x = int(elements_per_process)
        if self.MPI_x_coord < remaining_elements/2:
            # Give the remaining to the lowest ranked processes:
            self.n_elements_x += 2

        # x_min is the sum of the widths of the MPI tasks to our left:
        self.x_min = self.x_min_global + elements_per_process * self.element_width_x * self.MPI_x_coord
        # Including the extra elements that the low ranked tasks might have:
        if self.MPI_x_coord < remaining_elements/2:
            self.x_min += 2*self.MPI_x_coord*self.element_width_x
        else:
            self.x_min += remaining_elements * self.element_width_x
        self.x_max = self.x_min + self.element_width_x * self.n_elements_x

        # We only split along the x direction for now:
        self.n_elements_y = self.n_elements_y_global
        self.y_min = self.y_min_global
        self.y_max = self.y_max_global

        # The data we want to send left and right isn't in contiguous memory, so we
        # need to copy it into and out of temporary buffers:
        self.MPI_left_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_left_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_send_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)
        self.MPI_right_receive_buffer = np.zeros((self.Ny * self.n_elements_y), dtype=complex)

        # We need to tag our data in case there are exactly two MPI tasks, in
        # which case we need a way other than rank to distinguish between the
        # messages from the left and the right of that task:
        TAG_LEFT_TO_RIGHT = 0
        TAG_RIGHT_TO_LEFT = 1
        self.MPI_send_left = self.MPI_comm.Send_init(self.MPI_left_send_buffer,
                                                     self.MPI_rank_left, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_send_right = self.MPI_comm.Send_init(self.MPI_right_send_buffer,
                                                      self.MPI_rank_right, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_left = self.MPI_comm.Recv_init(self.MPI_left_receive_buffer,
                                                        self.MPI_rank_left, tag=TAG_LEFT_TO_RIGHT)
        self.MPI_receive_right = self.MPI_comm.Recv_init(self.MPI_right_receive_buffer,
                                                         self.MPI_rank_right, tag=TAG_RIGHT_TO_LEFT)
        self.MPI_all_requests = [self.MPI_send_left, self.MPI_send_right, self.MPI_receive_left, self.MPI_receive_right]
        self.MPI_left_to_right_requests = [self.MPI_send_right, self.MPI_receive_left]
        self.MPI_right_to_left_requests = [self.MPI_send_left, self.MPI_receive_right]

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
        the result"""
        # x kinetic energy operator operating on psi.
        Kx_psi = np.einsum('ij,xyjl->xyil', self.Kx,  psi)

        # Initiate MPI transfer for exchanging x kinetic energy contributions
        # now, will get back to this after computing y kinetic energy:
        self.MPI_left_send_buffer[:] = Kx_psi[0, :, 0, :].reshape(self.n_elements_y * self.Ny)
        self.MPI_right_send_buffer[:] = Kx_psi[-1, :, -1, :].reshape(self.n_elements_y * self.Ny)
        MPI.Prequest.Startall(self.MPI_all_requests)

        # Add contributions to x kinetic energy from both elements adjacent to each internal edge:
        Kx_psi[1:, :, 0, :] = Kx_psi[:-1, :, -1, :] = Kx_psi[1:, :, 0, :] + Kx_psi[:-1, :, -1, :]

        # y kinetic energy operator operating on psi.
        Ky_psi = np.einsum('kl,xyjl->xyjk', self.Ky,  psi)

        # Add contributions to y kinetic energy from both elements adjacent to each internal edge:
        Ky_psi[:, 1:, :, 0] = Ky_psi[:, :-1, :, -1] = Ky_psi[:, 1:, :, 0] + Ky_psi[:, :-1, :, -1]
        # Periodic boundary conditions in y direction:
        Ky_psi[:, 0, :, 0] = Ky_psi[:, -1, :, -1] = Ky_psi[:, 0, :, 0] + Ky_psi[:, -1, :, -1]

        # Add contributions to x kinetic energy from adjacent MPI tasks:
        MPI.Prequest.Waitall(self.MPI_all_requests)
        left_data = self.MPI_left_receive_buffer.reshape((self.n_elements_y, self.Ny))
        right_data = self.MPI_right_receive_buffer.reshape((self.n_elements_y, self.Ny))
        if psi.dtype is not float:
            # MPI buffers are complex, but the data is real, and we don't want
            # to emit a casting warning:
            left_data = left_data.real
            right_data = right_data.real
        Kx_psi[0, :, 0, :] += left_data
        Kx_psi[-1, :, -1, :] += right_data

        return Kx_psi + Ky_psi

    def compute_mu(self, psi, V, uncertainty=False):
        """Calculate chemical potential of DVR basis wavefunction psi with
        potential V"""

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

    def compute_energy(self, psi, V):
        """Calculate total energy of DVR basis wavefunction psi with
        potential V (same shape as psi)"""

        # Total kinetic energy operator operating on psi:
        K_psi = self.compute_K_psi(psi)

        density = psi.conj() * self.density_operator * psi

        # Total energy operator. Differs from total Hamiltonian in that the
        # nonlinear term is halved in order to avoid double counting the
        # interaction energy:
        E_total_psi = K_psi + (V + 0.5*self.g * density) * psi
        Ecalc = self.global_dot(psi, E_total_psi)
        return Ecalc

    def find_groundstate(self, psi_guess, V, mu, convergence=1e-14, relaxation_parameter=1.7,
                         callback_func=None, callback_period=100, print_period=100):
        """Find the groundstate of the given spatial potential V with chemical
        potential mu. Callback_func, if provided, will be called  every
        callback_period steps"""

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

        i = 0

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

        mucalc = self.compute_mu(psi, V)
        start_time = time.time()
        while True:
            # We operate on all elements at once, but only some DVR basis functions at a time.
            for points, j, k, in points_and_indices:
                if points is edges:
                    # We update all the edge points simultaneously.
                    # x kinetic energy operator operating on psi at the edge points:
                    Kx_psi[:, :, (0, -1), :] = np.einsum('ij,xyjl->xyil', Kx[(0, -1), :],  psi)
                    Kx_psi[:, :, :, (0, -1)] = np.einsum('ij,xyjl->xyil', Kx,  psi[:, :, :, (0, -1)])

                    # Initiate MPI transfer for exchanging x kinetic energy contributions
                    # now, will get back to this after computing y kinetic energy:
                    self.MPI_left_send_buffer[:] = Kx_psi[0, :, 0, :].reshape(self.n_elements_y * self.Ny)
                    self.MPI_right_send_buffer[:] = Kx_psi[-1, :, -1, :].reshape(self.n_elements_y * self.Ny)
                    MPI.Prequest.Startall(self.MPI_all_requests)

                    # Add contributions to x kinetic energy from both elements adjacent to each internal edge:
                    Kx_psi[1:, :, 0, :] = Kx_psi[:-1, :, -1, :] = Kx_psi[1:, :, 0, :] + Kx_psi[:-1, :, -1, :]

                    # y kinetic energy operator operating on psi at the edge points:
                    Ky_psi[:, :, (0, -1), :] = np.einsum('kl,xyjl->xyjk', Ky,  psi[:, :, (0, -1), :])
                    Ky_psi[:, :, :, (0, -1)] = np.einsum('kl,xyjl->xyjk', Ky[(0, -1), :],  psi)

                    # Add contributions to y kinetic energy from both elements adjacent to each internal edge:
                    Ky_psi[:, 1:, :, 0] = Ky_psi[:, :-1, :, -1] = Ky_psi[:, 1:, :, 0] + Ky_psi[:, :-1, :, -1]
                    # Periodic boundary conditions in y direction:
                    Ky_psi[:, 0, :, 0] = Ky_psi[:, -1, :, -1] = Ky_psi[:, 0, :, 0] + Ky_psi[:, -1, :, -1]

                    # Add contributions to x kinetic energy from adjacent MPI tasks:
                    MPI.Prequest.Waitall(self.MPI_all_requests)
                    Kx_psi[0, :, 0, :] += self.MPI_left_receive_buffer.reshape((self.n_elements_y, self.Ny)).real
                    Kx_psi[-1, :, -1, :] += self.MPI_right_receive_buffer.reshape((self.n_elements_y, self.Ny)).real
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

            if callback_func is not None and not i % callback_period:
                callback_func(i, psi)
            if not i % print_period:
                mucalc = self.compute_mu(psi, V)
                convergence_calc = abs((mucalc - mu)/mu)
                print(i, 'mu:', mu, 'mucalc:', mucalc,
                      'convergence:', convergence_calc,
                      round(1e3*(time.time() - start_time)/(i+1), 2), 'ms per step')
                if convergence_calc < convergence:
                    print('convergence reached')
                    if callback_func is not None and i % callback_period: # Don't call if already called this step
                        callback_func(i, psi)
                    break
            i += 1

        # Return complex array ready for time evolution:
        psi = np.array(psi, dtype=complex)
        return psi


    def evolve(self, psi, V, t_final, callback_func=None, callback_period=100,
               print_period=100, imaginary_time=False):

        dx_min = np.diff(self.x[0, 0, :, 0]).min()
        dy_min = np.diff(self.y[0, 0, 0, :]).min()
        dt = min(dx_min, dy_min)**2 * self.m / (2 * pi * hbar)
        print('using dt = ', dt)

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

        def copy_x_edges_odd_to_even(psi):
            """Copy odd endpoints -> adjacent even endpoints in x direction,
            including to and from adjacent MPI tasks:"""
            # Send values to the right and and receive from the left adjacent
            # processes. We're sending as early as possible and we'll deal with
            # the received data after doing other internal copying that we need to
            # do anyway (this minimises synchronisation overhead). The first
            # element in the x direction, number 0, is even, so we need to receive
            # into it but not send. The last element, number n_elements_x - 1, is
            # odd, so we need to send from it.

            # TODO: If there is still too much latency from this, split up the
            # einsum line in the main loop to do the edge elements first, then
            # initiate communication before doing einsum on all the internal
            # elements.
            self.MPI_right_send_buffer[:] = psi[-1, :, -1, :].reshape(self.n_elements_y * self.Ny)
            MPI.Prequest.Startall(self.MPI_left_to_right_requests)

            # Copy values from odd -> even elements at internal edges in x direction:
            psi[even_elements_x, :, -1, :] = psi[odd_elements_x, :, 0, :]
            psi[even_internal_elements_x, :, 0, :] = psi[odd_internal_elements_x, :, -1, :]

            # Now is the latest we can wait without the data from adjacent
            # processes. Wait for communication to complete:
            MPI.Prequest.Waitall(self.MPI_left_to_right_requests)
            # Copy over neighbouring process's value for psi at the boundary:
            psi[0, :, 0, :] = self.MPI_left_receive_buffer.reshape((self.n_elements_y, self.Ny))

        def copy_x_edges_even_to_odd(psi):
            """Copy even endpoints -> adjacent odd endpoints in x direction,
            including to and from adjacent MPI tasks:"""
            # Send values to the left and and receive from the right adjacent
            # processes. We're sending as early as possible and we'll deal with
            # the received data after doing other internal copying that we need to
            # do anyway (this minimises synchronisation overhead). The first
            # element in the x direction, number 0, is even, so we need to send
            # from it but not receive. The last element, number n_elements_x - 1, is
            # odd, so we need to receive into it.

            # TODO: If there is still too much latency from this, split up the
            # einsum line in the main loop to do the edge elements first, then
            # initiate communication before doing einsum on all the internal
            # elements.
            self.MPI_left_send_buffer[:] = psi[0, :, 0, :].reshape(self.n_elements_y * self.Ny)
            MPI.Prequest.Startall(self.MPI_right_to_left_requests)

            # Copy values from even -> odd elements at internal edges in x direction:
            psi[odd_internal_elements_x, :, -1, :] = psi[even_internal_elements_x, :, 0, :]
            psi[odd_elements_x, :, 0, :] = psi[even_elements_x, :, -1, :]

            # Now is the latest we can wait without the data from adjacent
            # processes. Wait for communication to complete:
            MPI.Prequest.Waitall(self.MPI_right_to_left_requests)
            # Copy over neighbouring process's value for psi at the boundary:
            psi[-1, :, -1, :] = self.MPI_right_receive_buffer.reshape((self.n_elements_y, self.Ny))

        def copy_y_edges_odd_to_even(psi):
            """Copy odd endpoints -> adjacent even endpoints in y direction. The y
            dimension is not split among MPI tasks, so there is no IPC here."""
            psi[:, even_elements_y, :, -1] = psi[:, odd_elements_y, :, 0]
            psi[:, even_internal_elements_y, :, 0] = psi[:, odd_internal_elements_y, :, -1]
            psi[:, 0, :, 0] = psi[:, -1, :, -1] # periodic boundary conditions

        def copy_y_edges_even_to_odd(psi):
            """Copy even endpoints -> adjacent odd endpoints in y direction. The y
            dimension is not split among MPI tasks, so there is no IPC here."""
            psi[:, odd_internal_elements_y, :, -1] = psi[:, even_internal_elements_y, :, 0]
            psi[:, odd_elements_y, :, 0] = psi[:, even_elements_y, :, -1]
            psi[:, -1, :, -1] = psi[:, 0, :, 0] # periodic boundary conditions

        if imaginary_time:
            # The kinetic energy unitary evolution oparators for half a timestep
            # in imaginary time, shapes (Nx, Nx) and (Ny, Ny). Not diagonal, but
            # the same in each element.
            U_Kx_halfstep = expm(-1/hbar * self.Kx * dt/2)
            U_Ky_halfstep = expm(-1/hbar * self.Ky * dt/2)
            # The same as above but for a full timestep:
            U_Kx_fullstep = expm(-1/hbar * self.Kx * dt)
            U_Ky_fullstep = expm(-1/hbar * self.Ky * dt)
        else:
            # The kinetic energy unitary evolution oparators for half a timestep,
            # shapes (Nx, Nx) and (Ny, Ny). Not diagonal, but the same in each
            # element.
            U_Kx_halfstep = expm(-1j/hbar * self.Kx * dt/2)
            U_Ky_halfstep = expm(-1j/hbar * self.Ky * dt/2)
            # The same as above but for a full timestep:
            U_Kx_fullstep = expm(-1j/hbar * self.Kx * dt)
            U_Ky_fullstep = expm(-1j/hbar * self.Ky * dt)

        # The potential energy evolution operator for the first half timestep. It
        # is always the same as at the end of timesteps, so we usually just re-use
        # at the start of each loop. But this being the first loop we need it now
        # too.
        density = (psi.conj()*self.density_operator*psi).real
        if imaginary_time:
            U_V_halfstep = np.exp(-1/hbar * (self.g * density + V - mu_initial) * dt/2)
        else:
            U_V_halfstep = np.exp(-1j/hbar * (self.g * density + V - mu_initial) * dt/2)

        i = 0
        t = 0
        start_time = time.time()
        while t < t_final:
            # Evolve for half a step with potential evolution operator:
            psi[:] = U_V_halfstep*psi

            # Evolution with x kinetic energy evolution operator, using odd-even-odd split step method:
            psi[odd_elements_x, :, :, :] = np.einsum('ij,xyjl->xyil', U_Kx_halfstep, psi[odd_elements_x, :, :, :])
            copy_x_edges_odd_to_even(psi)
            psi[even_elements_x, :, :, :] = np.einsum('ij,xyjl->xyil', U_Kx_fullstep, psi[even_elements_x, :, :, :])
            copy_x_edges_even_to_odd(psi)
            psi[odd_elements_x, :, :, :] = np.einsum('ij,xyjl->xyil', U_Kx_halfstep, psi[odd_elements_x, :, :, :])
            copy_x_edges_odd_to_even(psi)

            # Evolution with y kinetic energy evolution operator, using odd-even-odd split step method:
            psi[:, odd_elements_y, :, :] = np.einsum('kl,xyjl->xyjk', U_Ky_halfstep, psi[:, odd_elements_y, :, :])
            copy_y_edges_odd_to_even(psi)
            psi[:, even_elements_y, :, :] = np.einsum('kl,xyjl->xyjk', U_Ky_fullstep, psi[:, even_elements_y, :, :])
            copy_y_edges_even_to_odd(psi)
            psi[:, odd_elements_y, :, :] = np.einsum('kl,xyjl->xyjk', U_Ky_halfstep, psi[:, odd_elements_y, :, :])
            copy_y_edges_odd_to_even(psi)

            # Calculate potential energy evolution operator for half a step
            if imaginary_time:
                self.normalise(psi, n_initial)
            density[:] = (psi.conj()*self.density_operator*psi).real
            if imaginary_time:
                U_V_halfstep[:] = np.exp(-1/hbar * (self.g * density + V - mu_initial) * dt/2)
            else:
                U_V_halfstep[:] = np.exp(-1j/hbar * (self.g * density + V - mu_initial) * dt/2)

            # Evolve for half a timestep with potential evolution operator:
            psi[:] = U_V_halfstep*psi

            if callback_func is not None and not i % callback_period:
                if imaginary_time:
                    self.normalise(psi, n_initial)
                callback_func(i, t, psi)
            if not i % print_period:
                if imaginary_time and i % callback_period: # Don't renormalise if already done this step:
                    self.normalise(psi, n_initial)
                Ecalc = self.compute_energy(psi, V)
                ncalc = self.compute_number(psi)
                print('step:', i,
                      ' t =', round(t*1e6,1),'us',
                      ' number_err: %.02e'%abs(ncalc/n_initial-1),
                      ' energy_err: %.02e'%abs(Ecalc/E_initial-1),
                      ' time per step:', round(1e3*(time.time() - start_time)/(i+1), 2), 'ms')
            i += 1
            t += dt

        # t_final reached:
        if imaginary_time:
            self.normalise(psi, n_initial)
        if callback_func is not None and i % callback_period: # Don't call if already called on the last step:
            callback_func(i, t, psi)
        return psi




