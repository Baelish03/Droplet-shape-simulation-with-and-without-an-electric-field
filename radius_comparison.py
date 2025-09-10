import time
import numpy as np
import matplotlib.pyplot as plt
import drop_profile as dp
from constants import Constants as const

# SIMULATE WATER
if __name__ == "__main__":
    start_time = time.time()

    radius_array = []
    max_r = []
    max_z = []
    STEP = 500
    for i in range(20):
        drop = dp.Drop((i + 1) * STEP, 10000)
        drop.compute()
        radius_array.append(drop)
        max_z.append(abs(min(drop.z_coor)))
        max_r.append(abs(max(drop.r_coor)))
    print(time.time() - start_time)


    fig, ax = plt.subplots(1, 2)
    ax[0].grid(which="both")
    ax[1].grid(which="both")
    xarray = np.array(range(1, len(radius_array) + 1)) * STEP
    ax[0].plot(xarray, max_r,
               marker=".", color="b", linestyle="--")
    ax[0].set_xlabel("Raggio della sommità " + r"$R_0$" + " [m]")
    ax[0].set_ylabel("Raggio massimo " + r"$|r_{max}|$" + " [m]")

    ax[1].plot(xarray, max_z,
               marker=".", color="b", linestyle="--")
    ax[1].plot([xarray[0], xarray[-1]], [const.MAX_HEIGHT(), const.MAX_HEIGHT()],
               color="orange")
    ax[1].set_xlabel("Raggio della sommità " + r"$R_0$" + " [m]")
    ax[1].set_ylabel("Altezza massima " + r"$|z_{min}|$" + " [m]")
    
    plt.show()
