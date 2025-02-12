import numpy as np

def xyztorst(X, Y, Z):
    """
    Convert (X, Y, Z) coordinates in an equilateral tetrahedron
    to (r, s, t) coordinates in the standard reference tetrahedron.
    
    Parameters:
        X, Y, Z : array-like
            Coordinates of points in the equilateral tetrahedron.
    
    Returns:
        r, s, t : ndarray
            Coordinates in the standard reference tetrahedron.
    """
    # Define equilateral tetrahedron vertices
    v1 = np.array([-1, -1/np.sqrt(3), -1/np.sqrt(6)])
    v2 = np.array([ 1, -1/np.sqrt(3), -1/np.sqrt(6)])
    v3 = np.array([ 0,  2/np.sqrt(3), -1/np.sqrt(6)])
    v4 = np.array([ 0,  0,  3/np.sqrt(6)])
    
    # Compute transformation
    rhs = np.vstack([X, Y, Z]) - 0.5 * (v2 + v3 + v4 - v1).reshape(-1, 1)
    A = np.column_stack([(v2 - v1) / 2, (v3 - v1) / 2, (v4 - v1) / 2])
    
    RST = np.linalg.solve(A, rhs)
    
    return RST[0, :], RST[1, :], RST[2, :]
# Example usage
X = np.array([-1, 1, 0, 0])
Y = np.array([-1/np.sqrt(3), -1/np.sqrt(3), 2/np.sqrt(3), 0])
Z = np.array([-1/np.sqrt(6), -1/np.sqrt(6), -1/np.sqrt(6), 3/np.sqrt(6)])

r, s, t = xyztorst(X, Y, Z)
print("r:", r)
print("s:", s)
print("t:", t)
