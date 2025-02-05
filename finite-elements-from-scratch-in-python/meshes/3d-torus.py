import gmsh
import sys

gmsh.initialize(sys.argv)

cm = 1e-02;
a = 6 * cm;
b = 4 * cm;
Lc = 0.01;

# Create an ellipse using OpenCASCADE
ellipse = gmsh.model.occ.addTorus(0, 0, 0, a, b)

# Synchronize the model
gmsh.model.occ.synchronize();

# Generate a 2D mesh
gmsh.model.mesh.generate(2)

# Save the mesh as a .msh file
gmsh.write("3d-torus.msh")

# Run the GUI (optional)
gmsh.fltk.run()

gmsh.finalize()
