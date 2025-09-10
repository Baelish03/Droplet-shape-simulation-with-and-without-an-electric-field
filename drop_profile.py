import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as sp
from constants import Constants as const

class Drop():
    """Calculates and plots a generic droplet shape of a liquid,
        in contact with a fluid and a solid platform above
        in a gravitational field"""
    def __init__(self, upper_radius, n_points, contact_angle=np.radians(1)):
        self.upper_radius = upper_radius
        self.n_points = n_points
        self.theta = np.linspace(0, contact_angle, n_points, dtype=np.float64)
        self.r_coor = np.zeros(n_points, dtype=np.float64)
        self.z_coor = np.zeros(n_points, dtype=np.float64)

    def middle_function(self, theta_i, r_i, z_i):
        """Compute the integrand function far from bound"""
        return 2 / self.upper_radius - np.sin(theta_i) / r_i - z_i / const.CRITICAL_RADIUS()**2

    def initial_function(self, z_i):
        """Compute the integrand function near the bound"""
        return 1 / self.upper_radius - z_i / const.CRITICAL_RADIUS()**2

    def compute(self):
        """
        For each point:
            1) Check if it is near the bound. If yes, use initial function,
                otherwise use the middle one to calculate the integrand function
                using the i-1 informations;
            2) Integrate the functions;
            3) Repeat 1) with updated infos.
        Use references to make it faster
        """
        n_points = self.n_points
        theta = self.theta
        r_fun = np.zeros(n_points, dtype=np.float64)
        z_fun = np.zeros(n_points, dtype=np.float64)
        r_coor = self.r_coor
        z_coor = self.z_coor
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for i in range(0, n_points):
            if r_coor[i] == 0:
                denom_i = self.initial_function(z_coor[i-1])
                r_fun[i] = cos_theta[i] / denom_i
                z_fun[i] = -sin_theta[i] / denom_i
                r_coor[i] = sp(r_fun[:i+1], theta[:i+1])
                z_coor[i] = sp(z_fun[:i+1], theta[:i+1])
                denom_i1 = self.initial_function(z_coor[i])
                r_fun[i] = cos_theta[i] / denom_i1
                z_fun[i] = -sin_theta[i] / denom_i1
            else:
                denom_i = self.middle_function(theta[i], r_coor[i], z_coor[i])
                r_fun[i] = cos_theta[i] / denom_i
                z_fun[i] = -sin_theta[i] / denom_i
                r_coor[i] = sp(r_fun[:i+1], theta[:i+1])
                z_coor[i] = sp(z_fun[:i+1], theta[:i+1])
                r_fun[i] = cos_theta[i] / denom_i
                z_fun[i] = -sin_theta[i] / denom_i
            r_coor[i] = sp(r_fun[:i+1], theta[:i+1])
            z_coor[i] = sp(z_fun[:i+1], theta[:i+1])

    def output(self):
        """Prints values to file"""
        np.savetxt("output.txt", np.column_stack((self.r_coor, self.z_coor)), delimiter="\t")

    def plotting(self):
        """Plot droplet shape"""
        axis = plt.subplots(1, 1)[1]
        axis.spines['bottom'].set_position('zero')
        axis.spines['left'].set_position('zero')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.set_xlabel('r [m]', size=14, labelpad=-34, x=1.03)
        axis.set_ylabel('z [m]', size=14, labelpad=-41, y=.5, rotation=0)
        axis.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
        axis.set_xticks([max(self.r_coor)],
                        [str("%.2f"%max(self.r_coor * 100)) + r"$\ \cdot 10^{-2}$"])
        axis.set_yticks([min(self.z_coor)],
                        [str("%.2f"%min(self.z_coor * 100000) + r"$\ \cdot 10^{-5}$")])
        axis.plot((1), (0), marker='>', transform=axis.get_yaxis_transform(),
                  markersize=4, color='black', clip_on=False)
        axis.plot((0), (1), marker='^', transform=axis.get_xaxis_transform(),
                  markersize=4, color='black', clip_on=False)
        axis.plot(self.r_coor, self.z_coor, linewidth=1, color="cyan")
        axis.plot([min(self.r_coor), max(self.r_coor)], [min(self.z_coor), min(self.z_coor)],
                color="k")
        axis.fill_between(x=self.r_coor, y1=np.array(min(self.z_coor)), y2=self.z_coor,
                        color="cyan", alpha=.2)
        plt.show()


if __name__ == "__main__":
    start_time = time.time()
    water = Drop(2500, 10000, contact_angle=const.CONTACT_ANGLE())
    water.compute()
    #water.output()
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(const.MAX_HEIGHT())
    print(abs(max(water.r_coor)),"\t", abs(min(water.z_coor))) 
    water.plotting()
