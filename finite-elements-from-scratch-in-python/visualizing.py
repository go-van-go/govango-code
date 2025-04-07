import matplotlib.pyplot as plt
import numpy as np

# Load your data
slow_data = np.load('slow_data.npy')[:430]
fast_data = np.load('fast_data.npy')[:430]

# Define timestep (in seconds)
time_step = 0.000235571

# Create time arrays
time_slow = np.arange(len(slow_data)) * time_step*10
time_fast = np.arange(len(fast_data)) * time_step*10

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot fast wave speed data
ax1.plot(time_fast, fast_data, 
         marker='s', markersize=3, 
         linestyle='--', linewidth=1,
         color='red', label='Fast Wave Speed')
ax1.set_title('Fast Wave Speed, c = 3.0, density = 3.0, xyz = (0.5, 0.5, 0.3)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Pressure')  # Changed to Pressure
ax1.grid(True, alpha=0.3)

# Plot slow wave speed data
ax2.plot(time_slow, slow_data, 
         marker='o', markersize=4, 
         linestyle='-', linewidth=1,
         color='blue', label='Slow Wave Speed')
ax2.set_title('Slow Wave Speed, c = 1.0, density = 1.0, xyz = (0.5, 0.5, 0.1)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Pressure')  # Changed to Pressure
ax2.grid(True, alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()
