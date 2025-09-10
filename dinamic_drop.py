import numpy as np
from scipy.integrate import simpson as sm
from constants import Constants as const

class DinamicDrop():
    """
    Class for dynamic simulation of a water droplet in
    idrodynamic balance 
    """
    def __init__(self, drop_center_id, drop_perim_id, r_scale, z_scale,
                 drop_perim_float_points):
        """
        Parameters
        ----------
        drop_center_idx : tuple
            Indici del centro della goccia sulla griglia (r_index, z_index).
        drop_perimeter_idx : tuple
            Indici dei punti del perimetro della goccia sulla griglia (r_indices, z_indices).
        r_scale, z_scale : float
            Fattori di scala per convertire indici griglia in metri.
        potential : ndarray
            Matrice del potenziale elettrico sulla griglia.
        real_drop_perimeter : tuple of ndarray
            Coordinate reali del perimetro della goccia (r, z) in indici griglia.
        """
        self.r_scale, self.z_scale = r_scale, z_scale
        self.r_center_id = drop_center_id[0]
        self.z_center_id = drop_center_id[1]
        self.r_perim_id, self.z_perim_id = drop_perim_id
        self.phi = np.array([], dtype=np.float64)
        self.r_perim_pt, self.z_perim_pt = drop_perim_float_points

    def get_normal_vectors(self):
        """
        Get the vectors orthogonal to the surface perimeter

                Returns
        -------
        ndarray
            Array 2xN of the vectors orthogonal to the perimeter
        """
        r_perim_pt, z_perim_pt = self.r_perim_pt, self.z_perim_pt
        arc_length = np.cumsum(np.sqrt(np.diff(r_perim_pt)**2 + np.diff(z_perim_pt)**2))
        arc_length = np.insert(arc_length, 0, 0)
        dr_ds = np.gradient(r_perim_pt, arc_length, edge_order=2)
        dz_ds = np.gradient(z_perim_pt, arc_length, edge_order=2)
        normalization = (dr_ds ** 2 + dz_ds ** 2) ** (1/2)
        return np.stack((-dz_ds / normalization, dr_ds / normalization))

    def get_perimeter_gradient(self, potential):
        """
        Get the gradient values of the potential on the dropet perimeter.
            PAY ATTENTION: potential is dilatated, so electrid field need racalibration.

        Returns
        -------
        ndarray
            Array 2xN that contains the derivate of potential on the two axis
        for each point of the perimeter.
        """
        r_perim_id, z_perim_id = self.r_perim_id, self.z_perim_id
        r_perim_pt, z_perim_pt = self.r_perim_pt, self.z_perim_pt
        deltar = np.abs(r_perim_id - r_perim_pt)
        deltaz = np.abs(z_perim_id - z_perim_pt)

        r_grad, z_grad = np.gradient(potential)
        r_grad /= self.z_scale
        z_grad /= self.z_scale

        # Avoid tip-effect
        # Obviously, for construction, deltar is always 0 because electric field is valued on int values of r
        r_grad = (deltar * r_grad[r_perim_id, z_perim_id] +
                  (1 - deltar) * r_grad[r_perim_id + 1, z_perim_id])
        z_grad = (deltaz * z_grad[r_perim_id, z_perim_id] +
                  (1 - deltaz) * z_grad[r_perim_id, z_perim_id + 1])
        return np.stack((r_grad, z_grad))

    def p_problem(self, potential):
        """
        Calculate electric pressure.

        Returns
        -------
        ndarray
            Array della pressione elettrica lungo il perimetro.
        """
        normal_vectors = self.get_normal_vectors()
        perim_grad = self.get_perimeter_gradient(potential)
        projection = np.multiply(perim_grad, normal_vectors)
        return const.EPSILON0() / 2 * (projection[0] + projection[1]) ** 2

    def correction_pressure(self, init_volume):
        """
        Correction pressure for volume conservation.
        Volume is calculated with simpson integration.

        Parameters
        ----------
        initial_volume : float
            Initial volume of the droplet

        Returns
        -------
        float
            Uniform correction pressure
        """
        norm_z_pt = self.z_perim_pt - self.z_center_id
        volume = sm(norm_z_pt, self.r_perim_pt)
        print(volume / init_volume)
        return - (volume / init_volume - 1) * const.ATM_PRESSURE()

    def hydrostatic_pressure(self, norm_z):
        """
        Calcolate hydrostatic pressure due to droplet height

        Parameters
        ----------
        z_normalized : ndarray
            Normalized height

        Returns
        -------
        ndarray
            Hydrostatic pressure in each point
        """
        return const.WATER_DENSITY() * const.GRAVITY() * (np.max(norm_z) - norm_z)

    def surface_pressure(self, radius, radius_prime, phi):
        """
        Calcola la pressione dovuta alla tensione superficiale basata sulla curvatura.

        Parameters
        ----------
        radius : ndarray
            Raggio polare della goccia.
        radius_derivative : ndarray
            Derivata del raggio rispetto all'angolo phi.
        phi : ndarray
            Angolo polare dei punti.

        Returns
        -------
        ndarray
            Pressione superficiale.
        """
        radius_primeprime = np.gradient(radius_prime, phi, edge_order=2)
        numer = radius ** 2 + 2 * radius_prime ** 2 - radius * radius_primeprime
        denom = (radius ** 2 + radius_prime ** 2) ** (3/2)
        surface_pression = - const.SURFACE_TENSION() * np.abs(numer) / denom

        # Bounds condition
        surface_pression[0] = 0
        surface_pression[-1] = surface_pression[-2]
        return surface_pression

    def r_problem(self, electric_pressure, init_volume):
        """
        Calculate all the pressures:
            - "electric" is already calculated
            - "correction" one depends on volume, calculated with simpson
                in cartesian coordinates;
            - "height" one uses cartesian coordinates;
            - "surface" depends on curvature so it uses polar ones;

        Returns
        -------
        ndarray
            Total pressure in each point
        """
        norm_r_meters = (self.r_perim_pt - self.r_center_id) * self.r_scale
        norm_z_meters = (self.z_perim_pt - self.z_center_id) * self.z_scale

        radius = np.sqrt(norm_r_meters ** 2 + norm_z_meters ** 2 )
        self.phi = np.arctan2(norm_z_meters, norm_r_meters)
        radius_prime = np.gradient(radius, self.phi, edge_order=2)

        # do not scale correction pressure, the area is dilatated but also initial area is
        correction_pressure = np.full_like(electric_pressure, self.correction_pressure(init_volume))
        hydrostatic_pressure = self.hydrostatic_pressure(norm_z_meters)
        surface_pression = self.surface_pressure(radius, radius_prime, self.phi)
        return electric_pressure + correction_pressure + hydrostatic_pressure + surface_pression

    def temporal_evolution(self, total_pressure, delta_t, elapsed_t):
        """
        Temporal evolution using total pressure

        Parameters
        ----------
        total_pressure : ndarray
            Total pressure for each point
        delta_t : float
            Temporal increment factor
        elapsed_t : float
            Elapsed time

        Returns
        -------
        tuple
            new coordinates r e z of the perimeters
        float
            updated time
        """
        r_new = self.r_perim_pt.copy()
        z_new = self.z_perim_pt.copy()
        evol = total_pressure * delta_t
        r_new += evol * np.cos(self.phi)
        z_new += evol * np.sin(self.phi)

        # Last point correction to respect contact angle
        start_r = r_new[-2] * self.r_scale
        delta_z = (z_new[-2] - z_new[-1]) * self.z_scale
        r_new[-1] = (start_r + delta_z / np.tan(const.CONTACT_ANGLE()) ) / self.r_scale

        # Lower point
        z_new[0] = z_new[1]

        elapsed_t += delta_t
        return np.stack((r_new, z_new)), elapsed_t
