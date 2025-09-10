import numpy as np
from pypardiso import spsolve
import pyamg

class PotentialCalculation:
    """
    Class for solving a Poisson equation system using either
    an exact solver (Pardiso) or a faster approximate solver (AMG).

    Parameters
    ----------
    poisson_matrix : scipy.sparse matrix
        Discretized Poisson operator (sparse matrix).
    boundary_vector : ndarray
        Right-hand side vector containing boundary conditions.
    n_points : int
        Number of points per dimension (assuming a square grid).
    type : str, optional
        Solution method: "exact" (default) uses Pardiso direct solver,
        "fast" uses Algebraic Multigrid (AMG).
    """
    def __init__(self, poisson_matrix, boundary_vector, n_points, calc_type="exact"):
        self.poisson_matrix = poisson_matrix
        self.boundary_vector = boundary_vector
        self.n_points = n_points
        self.potential_solution = []

        # Choose solver type
        if calc_type == "exact":
            self.exact_calculation()
        elif calc_type == "fast":
            self.fast_calculation()

    def exact_calculation(self):
        """
        Solve the linear system exactly using Pardiso (direct solver).
        The solution vector is reshaped into a 2D grid of size (n_points, n_points).
        """
        potential_solution = spsolve(self.poisson_matrix, self.boundary_vector)
        potential_solution = potential_solution.reshape(self.n_points, self.n_points)
        self.potential_solution = potential_solution

    def fast_calculation(self):
        """
        Solve the linear system approximately using Algebraic Multigrid (AMG).
        Builds a multigrid hierarchy with Ruge-Stuben solver.
        Prints the solver hierarchy and residual norm for diagnostics.
        The solution vector is reshaped into a 2D grid of size (n_points, n_points).
        """
        multigrid_hierarchy = pyamg.ruge_stuben_solver(self.poisson_matrix)
        potential_solution = multigrid_hierarchy.solve(self.boundary_vector, tol=1e-15)

        # Compute residual to check accuracy
        print("residual: ", np.linalg.norm(self.boundary_vector
                                           - self.poisson_matrix * potential_solution))

        potential_solution = np.array(potential_solution).reshape(self.n_points, self.n_points)
        self.potential_solution = potential_solution
