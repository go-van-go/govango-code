import gmsh

def generate_structured_tet_mesh():
    gmsh.initialize()
    gmsh.model.add("structured_cube")

    # Create cube using 6 faces
    lc = 0.5  # Grid spacing (adjust as needed)
    
    p1 = gmsh.model.geo.addPoint(-1, -1, -1, lc)
    p2 = gmsh.model.geo.addPoint( 1, -1, -1, lc)
    p3 = gmsh.model.geo.addPoint( 1,  1, -1, lc)
    p4 = gmsh.model.geo.addPoint(-1,  1, -1, lc)
    p5 = gmsh.model.geo.addPoint(-1, -1,  1, lc)
    p6 = gmsh.model.geo.addPoint( 1, -1,  1, lc)
    p7 = gmsh.model.geo.addPoint( 1,  1,  1, lc)
    p8 = gmsh.model.geo.addPoint(-1,  1,  1, lc)

    # Define lines to form cube edges
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)

    l9  = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)

    # Create cube faces
    f1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.geo.addPlaneSurface([f1])

    f2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
    s2 = gmsh.model.geo.addPlaneSurface([f2])

    f3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
    s3 = gmsh.model.geo.addPlaneSurface([f3])

    f4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
    s4 = gmsh.model.geo.addPlaneSurface([f4])

    f5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
    s5 = gmsh.model.geo.addPlaneSurface([f5])

    f6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])
    s6 = gmsh.model.geo.addPlaneSurface([f6])

    # Create volume from surfaces
    sl = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
    volume = gmsh.model.geo.addVolume([sl])

    # Enforce transfinite meshing
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(s1)
    gmsh.model.mesh.setTransfiniteSurface(s2)
    gmsh.model.mesh.setTransfiniteSurface(s3)
    gmsh.model.mesh.setTransfiniteSurface(s4)
    gmsh.model.mesh.setTransfiniteSurface(s5)
    gmsh.model.mesh.setTransfiniteSurface(s6)
    gmsh.model.mesh.setTransfiniteVolume(volume)
    
    # Recombine into tetrahedra
    gmsh.model.mesh.generate(3)
    gmsh.write("structured_cube.msh")

    # Optional: Visualize mesh
    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    generate_structured_tet_mesh()
