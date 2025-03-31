// Define mesh size
lc = 0.1;  // Adjust this value to make the elements smaller

// Define points with a smaller characteristic length
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

Point(5) = {0, 0, 0.5, lc};
Point(6) = {1, 0, 0.5, lc};
Point(7) = {1, 1, 0.5, lc};
Point(8) = {0, 1, 0.5, lc};

Point(9) = {0, 0, 1, lc};
Point(10) = {1, 0, 1, lc};
Point(11) = {1, 1, 1, lc};
Point(12) = {0, 1, 1, lc};

// Define lower cube faces
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Define upper cube faces
Line(13) = {9, 10};
Line(14) = {10, 11};
Line(15) = {11, 12};
Line(16) = {12, 9};

Line(17) = {5, 9};
Line(18) = {6, 10};
Line(19) = {7, 11};
Line(20) = {8, 12};

// Define surfaces
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};

Line Loop(3) = {13, 14, 15, 16};
Plane Surface(3) = {3};

// Define side surfaces
Line Loop(4) = {1, 10, -5, -9};
Plane Surface(4) = {4};

Line Loop(5) = {2, 11, -6, -10};
Plane Surface(5) = {5};

Line Loop(6) = {3, 12, -7, -11};
Plane Surface(6) = {6};

Line Loop(7) = {4, 9, -8, -12};
Plane Surface(7) = {7};

Line Loop(8) = {5, 18, -13, -17};
Plane Surface(8) = {8};

Line Loop(9) = {6, 19, -14, -18};
Plane Surface(9) = {9};

Line Loop(10) = {7, 20, -15, -19};
Plane Surface(10) = {10};

Line Loop(11) = {8, 17, -16, -20};
Plane Surface(11) = {11};

// Define volumes
Surface Loop(1) = {1, 2, 4, 5, 6, 7};
Volume(1) = {1};

Surface Loop(2) = {2, 3, 8, 9, 10, 11};
Volume(2) = {2};

// Define physical groups
Physical Volume("LowerVolume") = {1};
Physical Volume("UpperVolume") = {2};

// Mesh settings
Mesh.ElementSizeFactor = 0.5;  // Make elements even smaller
Mesh 3;
