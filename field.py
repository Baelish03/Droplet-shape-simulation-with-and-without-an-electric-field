import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib import colors
import numpy as np
from constants import Constants as const


class Field():
    """
    Class to simulate the electic potential and build discrete
    Poisson matrix on a 2D meshgrid
    """
    def __init__(self, n_points, r_scale, z_scale):
        """
        Parameters
        ----------
        n_points : int
            Nomber od points of each edge of the grid (NxN).
        r_scale : float
            Scale factor for radial axis (r).
        z_scale : float
            Scale factor for vertical axis (z).
        """
        self.n_points = n_points
        self.r_scale = r_scale
        self.z_scale = z_scale

        self.r_grid, self.z_grid = np.meshgrid(np.arange(n_points), np.arange(n_points))
        self.boundary_potential = np.zeros((n_points, n_points))
        self.poisson_matrix = sp.csr_matrix((1,1))

    def poisson_sparse_matrix(self, base_height, epsilon_base=1, epsilon_air=1):
        """
        Construct the discrete Poisson matrix with spatially varying dielectric constants.

        This function builds a sparse (N^2 × N^2) matrix that represents the discretized
        Poisson equation on a 2D Cartesian grid with non-uniform dielectric permittivity.
        The resulting matrix gonna be inverted to compute the potential distribution 
        in the domain.

        The procedure follows these main steps:

        1. Dielectric map (node-centered):
        - Each grid point (node) is assigned a relative permittivity.
        - By default, the grid is initialized with `epsilon_air`.
        - All nodes below a given `base_height` are set to `epsilon_base`.
        - This creates a piecewise-constant dielectric distribution.

        2. Dielectric map (face-centered):
        - For finite-difference discretization, dielectric values must be
            defined not only at grid points (nodes) but also at the midpoints
            between neighboring nodes (faces).
        - A harmonic mean of adjacent node permittivities is used to compute
            effective dielectric constants at the faces. This ensures continuity
            of flux across material boundaries.
        - Vertical face values are stored in odd rows, even columns.
        - Horizontal face values are stored in even rows, odd columns.

        3. Finite-difference coefficients:
        - Using the face-centered permittivities, finite-difference coefficients
            for the four neighbors (left, right, up, down) are computed:
            * `coeff_left`, `coeff_right` scale the radial contributions.
            * `coeff_up`, `coeff_down` scale the vertical contributions.
        - These coefficients are normalized by the grid step sizes
            (`r_step` and `z_step`).

        4. Matrix assembly (interior points):
        - Each interior node is assigned a unique integer index (`point_id`).
        - For each interior node, contributions are added to the matrix:
            * The diagonal entry stores the negative sum of all neighbor couplings:
                `- (r_step2 * (coeff_left + coeff_right) - z_step2 * (coeff_up + coeff_down))`
            * Off-diagonal entries connect to the corresponding neighbor node
                indices with positive weights (`r_step2 * coeff_right`, etc.).
        - These entries are stored in coordinate (row, col, value) format.

        5. Boundary conditions:
        - Boundary nodes are enforced as Dirichlet conditions (fixed potential).
        - For each boundary node:
            * The matrix row is replaced with a unit row `[i, i] = 1`.
            * This ensures the potential at boundaries remains fixed to the
                boundary condition (imposed externally).

        6. Sparse matrix construction:
        - All row, column, and data arrays are concatenated.
        - A `scipy.sparse.csr_matrix` is built with shape `(N^2, N^2)`.
        - This matrix represents the discrete Poisson operator for the given
            dielectric configuration.

        Parameters
        ----------
        base_height : int
            Base height with epsilon_base
        epsilon_base : float
            Permittivity below the base
        epsilon_air : float
            Permittivity of the air over the base

        Returns
        -------
        poisson_matrix : scipy.sparse.csr_matrix
            The assembled sparse Poisson operator of size (N^2 × N^2).
            This can be inverted or used with iterative solvers to compute
                the electrostatic potential.

        Notes
        -------
            - To create a homogeneous medium, simply set `epsilon_air == epsilon_base`.
        """
        n_points = self.n_points
        total_points = n_points * n_points
        epsilon_nodes = np.full((n_points, n_points), epsilon_air, dtype=np.float64)
        epsilon_nodes[int(base_height):, :] = epsilon_base
        epsilon_faces = np.zeros((2 * n_points - 1, 2 * n_points - 1), dtype=np.float64)
        # vertical harmonic mean
        epsilon_faces[1: :2, : :2] = ( (2 * epsilon_nodes[0:-1, :] * epsilon_nodes[1:, :] /
                                        (epsilon_nodes[0:-1, :] + epsilon_nodes[1:, :])) *
                                        const.EPSILON0() )
        # orizzontal harmonic mean
        epsilon_faces[: :2, 1: :2] = ( (2 * epsilon_nodes[:, 0:-1] * epsilon_nodes[:, 1:] /
                                        (epsilon_nodes[:, 0:-1] + epsilon_nodes[:, 1:])) *
                                        const.EPSILON0() )

        coeff_down = epsilon_faces[3: :2, 2:-1:2].ravel() #odd rows, even columns
        coeff_up = epsilon_faces[1:-2:2, 2:-1:2].ravel()
        coeff_right = epsilon_faces[2:-1:2, 3: :2].ravel() #odd rows, even columns
        coeff_left = epsilon_faces[2:-1:2, 1:-2:2].ravel()

        point_id = np.arange(total_points, dtype=np.intp).reshape(n_points, n_points)
        center_id = point_id[1:-1, 1:-1].ravel()
        up_id = point_id[2:, 1:-1].ravel()
        down_id = point_id[:-2, 1:-1].ravel()
        right_id = point_id[1:-1, 2:].ravel()
        left_id = point_id[1:-1, :-2].ravel()

        data = np.concatenate([
            - (coeff_left + coeff_right + coeff_up + coeff_down),
            coeff_right, coeff_left, coeff_up, coeff_down])
        rows = np.concatenate([center_id] * 5)
        cols = np.concatenate([center_id, right_id, left_id, up_id, down_id])

        boundary_id = np.unique(np.concatenate([point_id[0, :], point_id[-1, :],
                                                 point_id[:, 0], point_id[:, -1]]))
        rows = np.concatenate([rows, boundary_id])
        cols = np.concatenate([cols, boundary_id])
        data = np.concatenate([data, np.full_like(boundary_id, 1.0)])
        self.poisson_matrix = sp.csr_matrix((data, (rows, cols)),
                                            shape=(total_points, total_points))
        return self.poisson_matrix

    def circular_electrode(self, potential, center_coor, radius):
        """
        Set an electod, whose became border condition, with a constant value for bound_func
        and the poisson matrix must be update to [i, i] = 1 and 0 in the rest of the row.
        
        Parameters
        ----------
        potential : float
            Electrode's voltage
        center_coords : tuple of int
            Coordinates (r, z) of the electrode's center
        radius : float
            Electrode's voltage     
        """
        r_center, z_center = center_coor
        radius2 = radius**2
        mask_circle = (self.r_grid - r_center)**2 + (self.z_grid - z_center)**2 <= radius2

        self.boundary_potential[mask_circle] = potential
        boundary_id = np.where(mask_circle.ravel())[0]

        n_points = self.n_points
        zeros = np.full_like(boundary_id, 0.0)
        rows = np.concatenate([boundary_id] * 5)
        cols = np.concatenate([boundary_id, boundary_id - 1, boundary_id + 1,
                               boundary_id + n_points, boundary_id - n_points])
        data = np.concatenate([np.full_like(boundary_id, 1.0), zeros, zeros, zeros, zeros])
        self.poisson_matrix[rows, cols] = data
        return self.boundary_potential

    def first_droplet(self, droplet_coords, base_center_coor, droplet_markers):
        """
        Map the first droplet on a (N x N) matrix using the chapter 1 data.
        Assign 1 to the coordinates of the droplet in the map and use
            it to change the poisson matrix as constant potential zone.

        Parameters
        ----------
        droplet_coords : tuple of arrays
            Real coordinates of the droplet (r_array, z_array) in meters
        base_center_coords : tuple of int
            Central coordinates of the droplet's base (r_index, z_index) on the grid.
        droplet_markers : list
            Droplet lenght in points [pt]

        Returns
        -------
        droplet_mask : ndarray
            NxN matrix with 1 inside the droplet and 0 otherwise
        droplet_grid_indices : ndarray
            Coordinates r and z of the droplet on the grid
        droplet_physical_coords : ndarray
            Coordinates r and z of the droplet in points
        """
        droplet_length_pts = int(droplet_markers[0])
        r_center_id, z_center_id = base_center_coor
        droplet_r, droplet_z = droplet_coords

        # Reparametrization of the curve
        r_resampled = np.linspace(np.min(droplet_r), np.max(droplet_r), droplet_length_pts)
        z_resampled = np.interp(r_resampled, droplet_r, droplet_z)
        z_points_float = (z_resampled + np.abs(np.min(z_resampled))) / self.z_scale + z_center_id

        # Indices on the grid
        r_indices_grid = np.intp(np.arange(0, droplet_length_pts) + r_center_id)
        z_indices_grid = np.array(z_resampled / self.z_scale, dtype=np.intp)
        z_indices_grid = np.intp(np.round(z_indices_grid + np.abs(np.min(z_indices_grid)),
                                          decimals=0) + z_center_id)

        n_points = self.n_points
        droplet_mask = np.full((n_points, n_points), fill_value=0, dtype=np.intp)
        for i, r_idx in enumerate(r_indices_grid):
            droplet_mask[int(z_center_id):z_indices_grid[i], r_idx] = 1

        # Update poisson matrix
        droplet_ids = np.where(droplet_mask.ravel() > 0)[0]
        zeros = np.zeros_like(droplet_ids)
        rows = np.concatenate([droplet_ids] * 5)
        cols = np.concatenate([droplet_ids, droplet_ids - 1, droplet_ids + 1,
                               droplet_ids + n_points, droplet_ids - n_points])
        data = np.concatenate([np.ones_like(droplet_ids), zeros, zeros, zeros, zeros])
        self.poisson_matrix[rows, cols] = data

        return np.stack((r_indices_grid, z_indices_grid)), np.stack((r_indices_grid, z_points_float))

    def droplet(self, droplet_coords, base_center_coor):
        """
        Map a subsquent droplet on the grid and update Poisson matrix

        Parameters
        ----------
        droplet_coords : tuple of arrays
            Coordinates of the droplet in points (not on the grid)
        base_center_coords : tuple of int
            Coordinatea of the droplet's center

        Returns
        -------
        droplet_mask : ndarray
            NxN matrix with 1 inside the droplet and 0 otherwise
        droplet_grid_indices : ndarray
            Coordinates r and z of the droplet on the grid
        droplet_physical_coords : ndarray
            Coordinates r and z of the droplet in points
        """
        droplet_r, droplet_z = droplet_coords
        z_base_id = base_center_coor[1]

        # Interpolation on the grid
        rmax_id = int(round(np.max(droplet_r))) #+ 1
        r_indices_grid = np.arange(0, rmax_id, 1, dtype=np.intp)
        z_indices_interp = np.interp(r_indices_grid, droplet_r, droplet_z)
        z_indices_interp[-1] = z_base_id
        z_points_float = z_indices_interp.copy()
        z_indices_grid = np.array(np.round(z_indices_interp, decimals=0), dtype=np.intp)

        n_points = self.n_points
        droplet_mask = np.full((n_points, n_points), fill_value=0, dtype=np.intp)
        for i, r_idx in enumerate(r_indices_grid):
            droplet_mask[int(z_base_id):z_indices_grid[i], r_idx] = 1

        # Update poisson matrix
        droplet_ids = np.where(droplet_mask.ravel() > 0)[0]
        zeros = np.zeros_like(droplet_ids)
        rows = np.concatenate([droplet_ids] * 5)
        cols = np.concatenate([droplet_ids, droplet_ids - 1, droplet_ids + 1,
                               droplet_ids + n_points, droplet_ids - n_points])
        data = np.concatenate([np.ones_like(droplet_ids), zeros, zeros, zeros, zeros])
        self.poisson_matrix[rows, cols] = data

        return np.stack((r_indices_grid, z_indices_grid)), np.stack((r_indices_grid, z_points_float))

    def potential_3d_plot(self, potential, center):
        """
        3D plot of the electric potential

        Parameters
        ----------
        potential : ndarray
            NxN matrix of electric potential
        center : tuple
            Grid translation (r, z)
        """
        fig = plt.figure()
        axis = fig.add_subplot(111, projection="3d")
        axis.set_aspect("auto")

        r_grid = (self.r_grid - int(center[0])) * self.r_scale
        z_grid = (self.z_grid - int(center[1])) * self.z_scale
        xmin, xmax = r_grid.min(), r_grid.max()
        ymin, ymax = z_grid.min(), z_grid.max()
        zmin, zmax = potential.min(), potential.max()
        axis.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

        surf = axis.plot_surface(r_grid, z_grid, potential, cmap = 'Spectral_r', alpha = 0.9)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        axis.contourf(r_grid, z_grid, potential, cmap = 'Spectral_r', offset = -1, alpha = 0.75)
        axis.contour(r_grid, z_grid, potential, colors = 'black', offset = -1)
        axis.contourf(r_grid, z_grid, potential, cmap = 'autumn_r',
                      offset = np.max(r_grid), zdir = 'x', alpha = 0.75)
        axis.contourf(r_grid, z_grid, potential, cmap = 'autumn_r',
                      offset = np.max(z_grid), zdir = 'y', alpha = 0.75)

        axis.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        axis.set_xlabel("Raggio (r) [m]")
        axis.set_ylabel("Altezza (z) [m]")
        axis.set_zlabel("Potenziale [V]")
        axis.view_init(elev=35, azim=135)
        axis.set_title("Potenziale elettrostatico")
        plt.tight_layout()
        plt.show()

    def field_stream_plot(self, potential, center):
        """
        Plot of the electric field stremlines

        Parameters
        ----------
        potential : ndarray
            NxN matrix of electric potential
        center : tuple
            Grid translation (r, z)
        """
        r_grid = (self.r_grid - int(center[0])) * self.r_scale
        z_grid = (self.z_grid - int(center[1])) * self.z_scale
        r_field, z_field = np.gradient(potential)
        r_field = r_field * self.r_scale
        z_field = z_field * self.z_scale
        xmin, xmax = r_grid.min(), r_grid.max()
        ymin, ymax = z_grid.min(), z_grid.max()

        fig, axis = plt.subplots()
        axis.set_xlim([xmin, xmax])
        axis.set_ylim([ymin, ymax])

        # Create a mask
        mask = np.zeros(r_field.shape, dtype=bool)
        r_mask = np.where(r_field == 0, True, False)
        z_mask = np.where(z_field == 0, True, False)
        speed = (r_field ** 2 + z_field ** 2) ** (1 / 2)
        mask = np.logical_and(r_mask, z_mask)
        r_field = np.ma.array(r_field, mask=mask)

        norm = colors.Normalize(vmin=speed.min(), vmax=speed.max() / 4)

        # Varying color along a streamline
        strm = axis.streamplot(r_grid, z_grid, r_field, z_field,
                               color=speed, linewidth=2, cmap='Spectral_r',
                               broken_streamlines=False, norm=norm)
        fig.colorbar(strm.lines, shrink=0.5, aspect=10, extend="max")
        axis.imshow(~mask, alpha=0.5, cmap='gray', aspect='auto', extent=(xmin, xmax, ymax, ymin))

        axis.set_xlabel("Raggio (r) [m]")
        axis.set_ylabel("Altezza (z) [m]")
        axis.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

        plt.tight_layout()
        plt.show()
