"""Dam break reference data"""

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Time evolution of the front (a) and the height (b) of a colapsing water column

time_front = np.array(
    [
        0.43,
        0.62,
        0.80,
        0.97,
        1.14,
        1.29,
        1.45,
        1.62,
        1.76,
        1.93,
        2.07,
        2.24,
        2.40,
        2.54,
        2.71,
        2.87,
        3.04,
        3.21,
        3.29,
        3.32,
    ]
)
y_front = np.array(
    [
        1.11,
        1.22,
        1.44,
        1.67,
        1.89,
        2.11,
        2.33,
        2.56,
        2.78,
        3.00,
        3.22,
        3.44,
        3.67,
        3.89,
        4.11,
        4.33,
        4.56,
        4.78,
        4.89,
        5.0,
    ]
)

time_height = np.array([0.0, 0.80, 1.29, 1.74, 2.15, 2.57, 3.08, 4.27, 6.29])
y_height = np.append(np.arange(1.0, 0.55, -0.11), np.arange(0.44, 0.0, -0.11))

fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 10))
axs[0].scatter(time_front, y_front, marker="^")
axs[0].set_xlim([0.0, 3.5])
axs[0].set_ylim([0, 6])
axs[0].set_xlabel("t[-]")
axs[0].set_ylabel("x_front / H")
axs[0].set_title("Time evolution of the front")

axs[1].scatter(time_height, y_height, marker="^")
axs[1].set_xlim([0.0, 3.5])
axs[1].set_ylim([0.0, 1.4])
axs[1].set_xlabel("t[-]")
axs[1].set_ylabel("h(x = 0) / H")
axs[1].set_title("Time evolution of the height")

now = datetime.datetime.now()
# plt.savefig(f"dambreak_data{now:%Y-%m-%d %H:%M:%S}.png")
plt.close()

pressure_profile = pd.read_csv("sph_data/Buchner_pressureProfile.csv")
# pressure_profile.columns = pressure_profile.iloc[0]
# pressure_profile = pressure_profile.drop(labels = 0)

x_val = pressure_profile.loc[:, "x"].values
y_val = pressure_profile.loc[:, " y"].values

fig2 = plt.figure()
plt.scatter(x_val, y_val, marker="^")
plt.xlim([0, 8])
plt.ylim([-0.2, 1.2])
plt.xlabel("t(g/H)^(1/2)")
plt.ylabel("p/rho*g*H")
plt.savefig(f"dambreak_data{now:%Y-%m-%d %H:%M:%S}.png")
