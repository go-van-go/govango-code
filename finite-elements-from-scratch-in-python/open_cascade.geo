SetFactory("OpenCASCADE");

x_dim = .20;
y_dim = .20;
z_dim = .20;
source_radius = 0.01;
main_cell_size = 0.015;
source_cell_size = 0.008;

Box(1) = {0, 0, 0, x_dim, y_dim, z_dim/2};
Box(2) = {0, 0, z_dim/2, x_dim, y_dim, z_dim/2};

BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; };
Circle(21) = {0.1, 0.1, 0, source_radius, 0, 2*Pi};
Curve Loop(21) = {21};
Plane Surface(21) = {21};
//BooleanUnion{Volume{1}}{Surface{21}}

BooleanFragments{ Volume{1}; Delete; }{ Surface{21}; };

MeshSize{ PointsOf{ Volume{1}; } } = main_cell_size;
MeshSize{ PointsOf{ Volume{2}; } } = main_cell_size;
MeshSize{ PointsOf{ Surface{21}; } } = source_cell_size;
// Define physical groups

Physical Volume("LowerVolume") = {1};
Physical Volume("UpperVolume") = {2};
Physical Surface("source") = {21};

// Mesh settings
Mesh 3;

Save "015_split_source.msh";
