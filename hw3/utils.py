import numpy as np



def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion
    
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    q = np.zeros(4)
    q[0] = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2 
    q[1] = (R[2,1] - R[1,2]) / (4 * q[0])
    q[2] = (R[0,2] - R[2,0]) / (4 * q[0])
    q[3] = (R[1,0] - R[0,1]) / (4 * q[0])
    return q



def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix
    
    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    
    R = np.zeros((3, 3))
    R[0,0] = 1 - 2*q[2]**2 - 2*q[3]**2
    R[0,1] = 2*q[1]*q[2] - 2*q[0]*q[3]
    R[0,2] = 2*q[1]*q[3] + 2*q[0]*q[2]

    R[1,0] = 2*q[1]*q[2] + 2*q[0]*q[3]
    R[1,1] = 1 - 2*q[1]**2 - 2*q[3]**2
    R[1,2] = 2*q[2]*q[3] - 2*q[0]*q[1]

    R[2,0] = 2*q[1]*q[3] - 2*q[0]*q[2]
    R[2,1] = 2*q[2]*q[3] + 2*q[0]*q[1]
    R[2,2] = 1 - 2*q[1]**2 - 2*q[2]**2

    return R