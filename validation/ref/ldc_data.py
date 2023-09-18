"""Lid-driven cavity reference data"""

import datetime

import matplotlib.pyplot as plt
import pandas as pd

u_vel = pd.read_csv("sph_data/ldc_data_u_vel.csv")
u_vel.columns = u_vel.iloc[0]

u_vel = u_vel.drop(labels=0)
u_vel["100"] = u_vel["100"].replace(
    ["0,00000"], "0.00000"
)  # had to make some corrections in the input data

y = (u_vel.loc[:, "y"].values).astype(float)
u_Re_100, u_Re_1000, u_Re_10000 = (
    (u_vel.loc[:, "100"].values).astype(float),
    (u_vel.loc[:, 1000.0].values).astype(float),
    (u_vel.loc[:, "10,000"].values).astype(float),
)

v_vel = pd.read_csv("sph_data/ldc_data_v_vel.csv")
v_vel.columns = v_vel.iloc[0]

v_vel = v_vel.drop(labels=0)  # had to make some corrections in the input data

x = (v_vel.loc[:, "x"].values).astype(float)
v_Re_100, v_Re_1000, v_Re_10000 = (
    (v_vel.loc[:, 100.0].values).astype(float),
    (v_vel.loc[:, 1000.0].values).astype(float),
    (v_vel.loc[:, "10,000"].values).astype(float),
)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, label="1")
ax2 = fig.add_subplot(111, label="2", frame_on=False)

ax.scatter(x, v_Re_100, color="C0", marker="s")
ax.set_xlabel("x", color="C0")
ax.set_ylabel("V_y(x)", color="C0")
ax.tick_params(axis="x", colors="C0")
ax.tick_params(axis="y", colors="C0")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([-0.6, 0.5])

ax2.scatter(u_Re_100, y, color="C1", marker="o")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel("V_x(y)", color="C1")
ax2.set_ylabel("y", color="C1")
ax2.xaxis.set_label_position("top")
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis="x", colors="C1")
ax2.tick_params(axis="y", colors="C1")
ax2.set_xlim([-0.4, 1.0])
ax2.set_ylim([0.0, 1.0])

now = datetime.datetime.now()
plt.savefig(f"ldc_data{now:%Y-%m-%d %H:%M:%S}.png")
