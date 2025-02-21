import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

# Define vertices of the tetrahedron
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Define edges as pairs of vertices
edges = [
    (vertices[0], vertices[1]),
    (vertices[0], vertices[2]),
    (vertices[0], vertices[3]),
    (vertices[1], vertices[2]),
    (vertices[1], vertices[3]),
    (vertices[2], vertices[3])
]

# Create a figure with a 3D axis
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("#eceff4")  # Set background color

# Convert edges to a Line3DCollection and add to plot
edge_collection = Line3DCollection(edges, colors='k', linewidths=2)
ax.add_collection3d(edge_collection)

# Plot vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="#5e81ac", s=1000)

# Set axis limits
ax.set_xlim([-.2, 1.2])
ax.set_ylim([-.2, 1.2])
ax.set_zlim([-.2, 1.2])

# Show axes but remove ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
