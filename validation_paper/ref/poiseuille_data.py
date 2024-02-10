"""Poiseuille flow reference data"""

import datetime

import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv("sph_data/rpf_t_2.csv")  # at t = 2
rpf_t_2_x = df1.loc[:, "x"].values
rpf_t_2_y = df1.loc[:, " y"].values
# plt.plot(rpf_t_2_x, rpf_t_2_y)

df2 = pd.read_csv("sph_data/rpf_t_10.csv")  # at t = 10
rpf_t_10_x = df2.loc[:, "x"].values
rpf_t_10_y = df2.loc[:, " y"].values

df3 = pd.read_csv("sph_data/rpf_t_20.csv")  # at t = 20
rpf_t_20_x = df3.loc[:, "x"].values
rpf_t_20_y = df3.loc[:, " y"].values

df4 = pd.read_csv("sph_data/rpf_t_100.csv")  # at t = 100
rpf_t_100_x = df4.loc[:, "x"].values
rpf_t_100_y = df4.loc[:, " y"].values

plt.plot(rpf_t_2_x, rpf_t_2_y, label="t = 2")
plt.plot(rpf_t_10_x, rpf_t_10_y, label="t = 10")
plt.plot(rpf_t_20_x, rpf_t_20_y, label="t = 20")
plt.plot(rpf_t_100_x, rpf_t_100_y, label="t = 100")
plt.legend()
plt.xlim(
    [
        0.0,
        1,
    ]
)
plt.ylim([0.0, 1.4])
now = datetime.datetime.now()
plt.savefig(f"poiseuille_data{now:%Y-%m-%d %H:%M:%S}.png")
