"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# TODO 1: read freefall.csv into arrays t and y
#         (hint: np.loadtxt with a comma delimiter, skipping the header)
data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)
t = data[:, 0]
y = data[:, 1]

# TODO 2: compute velocity v = derivative of y w.r.t. t   (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?
v = np.gradient(y, t)
a = np.gradient(v, t)

mean = np.mean(a)
print(f"Mean of acceleration is {mean}")

std = a.std()
print(f"Standart deviation of acceleration is {std}")
# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)
recoveredV = integrate.cumulative_trapezoid(a, t, initial=0) + v[0]
recoveredY = integrate.cumulative_trapezoid(recoveredV, t, initial=0) + y[0]

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png
fig, axes = plt.subplots(1, 3, sharex = True, sharey = True)
axes[0].scatter(t, recoveredY)
axes[0].set_title("Recovered Position")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Count")
axes[1].scatter(t, recoveredV)
axes[1].set_title("Recovered Velocity")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Count")
axes[2].scatter(t, a)
axes[2].set_title("      Calculated acceleration")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Count")
axes[2].axhline(y=-9.18, linestyle="--")

plt.savefig("motion.png")

# Bonus Part
data = np.loadtxt("trajectory.csv", delimiter=",", skiprows=1)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

vX = np.gradient(x, t)
vY = np.gradient(y, t)

vFinals = ((vX**2)+(vY**2))**0.5

fig, axes = plt.subplots(1, 2, sharex = True, sharey = True)
axes[0].scatter(x, y)
axes[0].set_title("Trajectory")
axes[0].set_xlabel("X Coordinate")
axes[0].set_ylabel("Y Coordinate")
axes[1].scatter(t, vFinals)
axes[1].set_title("Recovered Velocity")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Speed")

plt.savefig("trajectory.png")