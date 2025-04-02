import numpy as np
import matplotlib.pyplot as plt
import time
from wave_simulator.mesh import Mesh3d
from wave_simulator.finite_elements import LagrangeElement

for i, point in enumerate(symmetric_points):
    symmetric_values[i, col_index] = visualizer.eval_at_point(point[0], point[1], point[2], physics.p)[0]
col_index += 1

symmetric_values = np.zeros((6, time_stepper.num_time_steps))
symmetric_points = [
    [0.25, 0.5, 0.5],
    [0.75, 0.5, 0.5],
    [0.5, 0.25, 0.5],
    [0.5, 0.75, 0.5],
    [0.5, 0.5, 0.25],
    [0.5, 0.5, 0.75]]


symmetric_values = np.load("symmetric_values.npy")
time_step = 0.0020202
time = np.arange(0, 495 * time_step, time_step)  # Create time axis

# Define different markers for each line
markers = ['o', 's', 'D', '^', 'v', 'x']  # Circle, square, diamond, triangle up, triangle down, cross

# Plot
plt.figure(figsize=(8, 5))
for i in range(6):
    plt.plot(time, symmetric_values[i], marker=markers[i], markersize=4, label=f'Line {i+1}')

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Line Plots of Symmetric Values')
plt.legend()
plt.grid(True)
plt.show()


