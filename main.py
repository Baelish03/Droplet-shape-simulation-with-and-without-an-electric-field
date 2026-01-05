"""
Main driver for droplet simulation in an external electric field.

Steps:
- Load initial droplet shape.
- Define computational domain and scaling factors. Set number 
    of points, then calculate the dimensions of the droplet and
    and set 250 intervals for height and 800 for width.
- Build field and electrodes.
- Iterate solving Poisson + droplet dynamics until convergence.
"""

import time
import numpy as np
from scipy.integrate import simpson as sm
import matplotlib.pyplot as plt
import matplotlib
from field import Field
from constants import Constants as const
from dinamic_drop import DinamicDrop
from potential_calculation import PotentialCalculation
np.set_printoptions(threshold=np.inf)

matplotlib.rcParams.update({'font.size': 24})

def compute_initial_volume(drop_perim_pt):
    """
    Calculate the volume of the droplet using simpson
        method on the perimeter.
    Perimeter must be shifted to have the lower point in 0

    Parameters
    ----------
    drop_perimeter : tuple of np.ndarray
        (r_values, z_values) defining the droplet perimeter.
        z_values are shifted so the lowest point is at z = 0.
    
    Returns
    -------
    float
        Droplet volume estimate.
    """
    r_perim_pt, z_perim_pt = drop_perim_pt
    z_perim_pt = z_perim_pt.copy()
    z_perim_pt -= np.min(z_perim_pt)
    return sm(z_perim_pt, r_perim_pt)

def plot_evolution(initial_shape, prev_perimeter, new_perimeter,
                   r_scale, z_scale, block_bool):
    """
    Plot the droplet evolution between iterations.
    """
    axis = plt.subplots(1, 1)[1]
    axis.spines['bottom'].set_position('zero')
    axis.spines['left'].set_position('zero')
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_xlabel('r [mm]', labelpad=-20, x=.5)
    axis.set_ylabel('z [' + r'$\mu$' + 'm]', labelpad=-30, y=.5, rotation=90)
    axis.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    initial_shape_local_r = initial_shape[0] - np.abs(np.min(initial_shape[0]))
    initial_shape_local_z = initial_shape[1] + np.abs(np.min(initial_shape[1]))
    new_perimeter_local_r = (new_perimeter[0] - np.abs(np.min(new_perimeter[0]))) * r_scale
    new_perimeter_local_z = (new_perimeter[1] - np.abs(np.min(new_perimeter[1]))) * z_scale
    axis.set_xticks([np.max(initial_shape_local_r), np.max(new_perimeter_local_r)],
                    [str("%.2f"%np.max(initial_shape_local_r * 100)),
                     str("%.2f"%np.max(new_perimeter_local_r * 100))])
    axis.set_yticks([np.max(initial_shape_local_z), np.max(new_perimeter_local_z)],
                    [str("%.2f"%np.max(initial_shape_local_z * 100000)),
                     str("%.2f"%np.max(new_perimeter_local_z * 100000))])
    axis.plot((1), (0), marker='>', transform=axis.get_yaxis_transform(),
                markersize=4, color='black', clip_on=False)
    axis.plot((0), (1), marker='^', transform=axis.get_xaxis_transform(),
                markersize=4, color='black', clip_on=False)

    axis.plot(initial_shape_local_r, initial_shape_local_z,
             color="b", marker=".", label="Goccia iniziale")
    #plt.plot(prev_perimeter[0], prev_perimeter[1],
    #         color="g", marker=".", label="Previous perimeter")
    axis.plot(new_perimeter_local_r, new_perimeter_local_z,
             color="r", marker=".", label="Goccia finale")

    axis.fill_between(x=initial_shape_local_r, y1=initial_shape_local_z,
                    color="b", alpha=.2)
    axis.fill_between(x=new_perimeter_local_r, y1=new_perimeter_local_z,
                    color="r", alpha=.2)
    axis.legend(loc="upper right")
    plt.show(block=block_bool)
    plt.pause(2)
    plt.close()


if __name__ == "__main__":
    import pypardiso as ppd
    raw_drop_data = np.loadtxt("output.txt", delimiter="\t")
    START_TIME = time.time()

    N_POINTS = 1000
    r_data, z_data = raw_drop_data[:, 0], raw_drop_data[:, 1]
    raw_drop_data = np.stack((r_data, z_data))

    DROP_WIDTH = np.max(r_data)
    DROP_HEIGHT = np.abs(np.min(z_data))

    # marker points used to rescale geometry
    DROP_MARKERS = (0.12 * N_POINTS - 1, 0.08 * N_POINTS - 1)
    R_SCALE = DROP_WIDTH / DROP_MARKERS[0]
    Z_SCALE = DROP_HEIGHT / DROP_MARKERS[1]

    ELECTRODE_POTENTIAL = 18e3 # V
    DROP_CENTER_ID = (0 * N_POINTS, 0.8 * N_POINTS)
    ELECTRODES_RADIUS = 0.15 * N_POINTS

    DELTA_TIME = 1e-4 # s
    #########################################################
    # Voltage   | Temporal step     | Relaxation factor     |
    # ----------+-------------------+-----------------------+
    # 1000      | e-3               |   1                   |
    # 5000      | e-3               |   1                   |
    # 10000     | 2e-4              |   1                   |
    # 15000     | e-4               |   1                   |
    # 18000     | e-4               |   1                   |
    #########################################################
    ELAPSED_TIME = 0 # s
    RELAXATION_FACTOR = 1
    CYCLE = 1
    CONVERGENCE_THRESHOLD = .84

    new_drop_perim_pt = []
    current_drop_perim_pt = []

    # --- Iterative loop ---
    while True:
        if CYCLE != 1:
            # Check convergence between iterations
            displacement2 = (new_drop_perim_pt - current_drop_perim_pt) ** 2
            displacement2 = displacement2[0] + displacement2[1]
            condition = displacement2 <= (1 ** 2)
            print("condition ", np.mean(condition))
            if np.mean(condition) >= CONVERGENCE_THRESHOLD:
                max_width = np.abs(np.max(new_drop_perim_pt[0])
                                   - np.min(new_drop_perim_pt[0])) * R_SCALE * 1e3
                max_height = np.abs(np.max(new_drop_perim_pt[1])
                                    - np.min(new_drop_perim_pt[1])) * Z_SCALE * 1e6
                print(f"End-time \t\t\t {time.time() - START_TIME:.2f} seconds")
                print(ELECTRODE_POTENTIAL, "\t", ELAPSED_TIME, "\t", max_width,
                      "\t", max_height, "\t", RELAXATION_FACTOR)
                plot_evolution(raw_drop_data, current_drop_perim_pt,
                               new_drop_perim_pt, R_SCALE, Z_SCALE,
                               block_bool=True)
                break

        field = Field(N_POINTS, R_SCALE, Z_SCALE)
        field.poisson_sparse_matrix(DROP_CENTER_ID[1], const.GLASS_REL_DIEL(), const.AIR_REL_DIEL())
        print(f"Matrix built \t\t\t {time.time() - START_TIME:.2f} seconds")

        field.circular_electrode(ELECTRODE_POTENTIAL,
                                 (0.7 * N_POINTS, 0.3 * N_POINTS),
                                 ELECTRODES_RADIUS)

        # --- Place droplet ---
        if CYCLE != 1:
            drop_indices, current_drop_perim_pt = field.droplet(new_drop_perim_pt, DROP_CENTER_ID)
        else:
            drop_indices, current_drop_perim_pt = field.first_droplet(raw_drop_data,
                                                                      DROP_CENTER_ID,
                                                                      DROP_MARKERS)
            initial_volume = compute_initial_volume(current_drop_perim_pt)
        print(f"Electrodes and droplet added \t {time.time() - START_TIME:.2f} seconds")

        poisson_matrix = field.poisson_matrix
        boundary_vector = field.boundary_potential.ravel()
        potential_calculation = PotentialCalculation(poisson_matrix, boundary_vector,
                                                   N_POINTS, calc_type="fast")
        potential_solution = potential_calculation.potential_solution
        print(f"Solved \t\t\t\t {time.time() - START_TIME:.2f} seconds")

        #field.potential_3d_plot(potential_solution, center=DROP_CENTER_ID)
        #field.field_stream_plot(potential_solution, center=DROP_CENTER_ID)

        droplet = DinamicDrop(DROP_CENTER_ID, drop_indices, R_SCALE, Z_SCALE, current_drop_perim_pt)
        electric_pressure = droplet.p_problem(potential_solution)
        print(f"P-Problem \t\t\t {time.time() - START_TIME:.2f} seconds")

        total_pressure = droplet.r_problem(electric_pressure, initial_volume)
        new_drop_perim_pt, ELAPSED_TIME = droplet.temporal_evolution(total_pressure,
                                                                     DELTA_TIME,
                                                                     ELAPSED_TIME)

        # Relax update
        new_drop_perim_pt = (RELAXATION_FACTOR * new_drop_perim_pt +
                             (1 - RELAXATION_FACTOR) * current_drop_perim_pt)
        print(f"R-Problem \t\t\t {time.time() - START_TIME:.2f} seconds")

        CYCLE += 1
        #plot_evolution(raw_drop_data, current_drop_perim_pt, new_drop_perim_pt,
        #               R_SCALE, Z_SCALE, block_bool=False)
