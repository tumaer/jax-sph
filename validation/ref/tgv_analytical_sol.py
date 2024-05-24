"""Taylor-Green vortex reference data"""

import datetime

import matplotlib.pyplot as plt
import numpy as np

key = np.random.default_rng(123).random()
box_size = np.array([1.0, 1.0])
dx = 0.05
std = dx / 8
n = np.array(box_size / dx, dtype=int)
grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
noise = std * np.random.normal(key, r.shape)
x, y = (r + noise).T
Re = 100  # Reynolds Number

### Time step calculation

rho_ref = 1.00
viscosity = 0.01
u_ref = 1.0
gamma_eos = 1.0

# Derived: reference speed of sound
c_ref = 10.0 * u_ref

# calculate volume and mass
h = dx
volume_ref = h**2
mass_ref = volume_ref * rho_ref

# time integration step dt
CFL = 0.25
dt_convective = CFL * h / (c_ref + u_ref)
dt_viscous = CFL * h**2 * rho_ref / viscosity
dt = np.amin([dt_convective, dt_viscous])

print("dt_convective :", dt_convective)
print("dt_viscous    :", dt_viscous)
print("dt            :", dt)

time = np.arange(0, 6, dt)
b = -8 * (np.pi * np.pi / Re)  # decay rate of velocity field
y_axes = np.empty(time.shape)
for i in np.arange(len(time)):
    u = -1.0 * np.exp(b * time[i]) * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    v = +1.0 * np.exp(b * time[i]) * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    u_abs = np.sqrt(u**2 + v**2)
    y_axes[i] = np.amax(u_abs)

fig, ax = plt.subplots()
ax.plot(time, y_axes, "--")
ax.set_yscale("log")
now = datetime.datetime.now()
plt.savefig(f"tgv_data{now:%Y-%m-%d %H:%M:%S}.png")
