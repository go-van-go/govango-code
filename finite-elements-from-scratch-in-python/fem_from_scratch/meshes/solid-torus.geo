// Parameters
r1 = 1.0;   // Minor radius
r2 = 2.5;   // Major radius
lc = 0.1;   // Mesh characteristic length

// Define torus using built-in function
SetFactory("OpenCASCADE");
Torus(1) = {0, 0, 0, r2, r1};

// Ensure the volume is properly recognized
BooleanFragments{ Volume{1}; Delete; }

// Mesh settings
Mesh.CharacteristicLengthMax = lc;
Mesh.CharacteristicLengthMin = lc / 2;
Mesh 3;  // Generate tetrahedral mesh

// Save mesh
Save "solid-torus.msh";
