import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.abspath(".."))  # Adds parent directory to sys.path

from recursivenodes import recursive_nodes

# Define polynomial orders and dimensions
orders = [1, 2, 3]
dimensions = [1, 2, 3]

# Create figure
fig = plt.figure(figsize=(5, 5))

for i, n in enumerate(orders):
    for j, d in enumerate(dimensions):
        ax_index = i * len(dimensions) + j + 1
        
        if d == 3:
            ax = fig.add_subplot(len(orders), len(dimensions), ax_index, projection='3d')
        else:
            ax = fig.add_subplot(len(orders), len(dimensions), ax_index)

        # Get nodes
        nodes = recursive_nodes(d, n, domain='unit')

        if d == 1:
            # 1D line segment
            ax.scatter(nodes[:, 0], np.zeros_like(nodes[:, 0]), c='#5e81ac', marker='o', s=50)
            ax.plot([0, 1], [0, 0], 'k-')
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.axis('off')

        elif d == 2:
            # 2D triangle
            ax.scatter(nodes[:, 0], nodes[:, 1], c='#5e81ac', marker='o', s=50)
            ax.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k-')  # Triangle edges
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        elif d == 3:
            # 3D tetrahedron
            ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='#5e81ac', marker='o', s=50)

            # Tetrahedron edges
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            
            for edge in edges:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                ax.scatter([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color='k', s=10)
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-')

            ax.set_xlim(0.1, 0.9)
            ax.set_ylim(0.1, 0.9)
            ax.set_zlim(0.1, 0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            # Make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            # Make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.grid(False)
            ax.set_axis_off()

            # Adjust the view to zoom in slightly
            ax.view_init(elev=20, azim=10)  # Modify elevation and azimuth for better visualization
            ax.dist = 3  # Decrease distance to zoom in

plt.tight_layout()
plt.show()
