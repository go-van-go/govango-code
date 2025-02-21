cm = 1e-02; // centimeters
a = 6 * cm; // Semi-major axis
b = 4 * cm; // Semi-minor axis
Lc = 0.01;  // characteristic cell length 

// Define center and ellipse points
Point(1) = {0, 0, 0, Lc};
Point(2) = {a, 0, 0, Lc};
Point(3) = {0, b, 0, Lc};
Point(4) = {-a, 0, 0, Lc};
Point(5) = {0, -b, 0, Lc};

// Define ellipse using center and points
Ellipse(1) = {2, 1, 3};
Ellipse(2) = {3, 1, 4};
Ellipse(3) = {4, 1, 5};
Ellipse(4) = {5, 1, 2};

// Define surface
Curve Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
