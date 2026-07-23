import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    max_r = [27.26, 27.28, 27.31, 27.35, 27.41, 27.48, 28, 28.62, 29.13, 29.75, 29.76, 30.40, 29.98, 30.25, 30.36, 31, 31.35, 30.63] #mm
    V = []

    for i in range(18):
        V.append((i+1)*1000)

    fig, ax = plt.subplots()
    ax.grid(which="both")
    ax.scatter(V[:7], max_r[:7],
               marker="o", color="b", label=r"$\omega = 1$")
    ax.scatter(V[7:10], max_r[7:10],
               marker="o", color="r", label=r"$\omega = 0.9$")
    ax.scatter(V[10:12], max_r[10:12],
               marker="o", color="g", label=r"$\omega = 0.75$")
    ax.scatter(V[12:15], max_r[12:15],
               marker="o", color="m", label=r"$\omega = 0.6$")
    ax.scatter(V[15:17], max_r[15:17],
               marker="o", color="k", label=r"$\omega = 0.5$")
    ax.scatter(V[17], max_r[17],
               marker="o", color="c", label=r"$\omega = 0.4$")
    ax.set_xlabel("Tensione degli elettrodi " + r"$V$" + " [V]")
    ax.set_ylabel("Raggio massimo " + r"$|r_{max}|$" + " [mm]")
    ax.legend(loc="upper left")
    plt.show()
