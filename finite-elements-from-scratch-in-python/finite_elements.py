import numpy as np
from pathlib import Path


class LagrangeElement:
    """Lagrange finite element defined on a triangle and
      tetrahedron"""
    # Get the base directory where the script is located, including the 'tabulated_nodes' folder
    base_dir = Path(__file__).parent / "tabulated_nodes"

    # Load the nodes once as a class attribute
    _line_nodes = np.load(base_dir / "line_nodes.npz")
    _triangle_nodes = np.load(base_dir / "triangle_nodes.npz")
    _tetrahedron_nodes = np.load(base_dir / "tetrahedron_nodes.npz")

    def __init__(self, d, n):
        self.d = d  # dimension
        self.n = n  # polynomial order
        self.nodes = np.array([])
        self.vertices = np.array([])
        self.nodesPerElement = (n + 1) * (n + 2) * (n + 3) // 6
        self.nodesPerFace = (n + 1) * (n + 2) // 2
        self.num_faces = 0  # no. faces per element
        self.NODETOL = 1e-7  # tolerance to find face nodes

        # Load verticies and precomputed nodes
        if d == 1 and n <= 30:
            # line element
            self.num_faces = 2
            self.nodes = self._line_nodes[str(n)]
            self.vertices = np.array([0,1])

        if d == 2 and n <= 30:
            # triangle
            self.num_faces = 3
            self.nodes = self._triangle_nodes[str(n)]
            self.vertices = np.array([[0, 0],
                                      [1, 0],
                                      [0, 1]])
        if d == 3 and n <= 30:
            # tetrahedron
            self.num_faces = 4
            self.nodes = self._tetrahedron_nodes[str(n)]
            self.vertices = np.array([[0, 0, 0],
                                      [1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
        else:
            raise Exception(f"No precomputed nodes found for d={d}, n={n}")

if __name__ == "__main__":
    pass
