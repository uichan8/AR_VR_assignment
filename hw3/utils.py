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

def jacobian_AB(A, B):
    """
    행렬 곱셈 C = A @ B에 대한 자코비안 행렬을 계산합니다.
    
    Parameters:
        A: 입력 행렬 A (m x n)
        B: 입력 행렬 B (n x p)
        
    Returns:
        자코비안 행렬 J (mp x mn 크기의 2차원 배열)
    """
    m, n = A.shape
    n, p = B.shape
    J = np.zeros((m * p, m * n))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                # C_ij = sum_k A_ik B_kj 이므로 ∂C_ij/∂A_ik = B_kj
                J[i * p + j, i * n + k] = B[k, j]
                
    return J

def jacobian_BA(B, A):
    """
    행렬 B와 A의 곱셈에 대한 자코비안 행렬을 계산합니다.

    Parameters:
        B: 입력 행렬 B (p x m)
        A: 입력 행렬 A (m x n)

    Returns:
        자코비안 행렬 J (p*n x m*n 크기의 2차원 배열)
    """
    p, m = B.shape
    m, n = A.shape
    J = np.zeros((p * n, m * n))
    
    for i in range(p):
        for j in range(n):
            for k in range(m):
                J[i * n + j, k * n + j] = B[i, k]
                
    return J

def quaternion_to_rotation_matrix_jacobian(q):
    w, x, y, z = q
    J = np.array([
        [0, 0, -4*y, -4*z],
        [-2*z, 2*y, 2*x, -2*w],
        [2*y, 2*z, 2*w, 2*x],
        
        [2*z, 2*y, 2*x, 2*w],
        [0, -4*x, 0, -4*z],
        [-2*x, 2*w, 2*z, -2*y],
        
        [-2*y, 2*z, 2*w, -2*x],
        [2*x, 2*w, 2*z, 2*y],
        [0, -4*x, -4*y, 0]
    ])
    return J

def ComputeReprojection(p, X):
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)
    _C = np.eye(3,4)
    _C[:3,3] = -C
    _X = np.hstack((X,1))
    B = R@_C@_X
    _B = B/B[2]
    return _B[:2]

def cal_reprojection_error(R,C,X,x):
    _C = np.eye(3,4)
    _C[:3,3] = -C
    _X = np.hstack((X,np.ones((X.shape[0],1)))).T
    B = R@_C@_X
    _B = B/B[2]
    _B = _B.T
    _B = _B[:,:2]

    errors = np.linalg.norm(_B - x, axis=1)
    return errors

def get_matching_from_track(track1,track2):
    valid_track1 = np.bitwise_not(track1 == -1)
    valid_track1 = np.bitwise_or(valid_track1[:,0],valid_track1[:,1])
    valid_track2 = np.bitwise_not(track2 == -1)
    valid_track2 = np.bitwise_or(valid_track2[:,0],valid_track2[:,1])
    valid_track = np.bitwise_and(valid_track1,valid_track2)
    _track1 = track1[valid_track]
    _track2 = track2[valid_track]
    valid_track = np.where(valid_track)[0]

    return _track1, _track2, valid_track
    
def decompose_extrinsic_matrix(P):
    """
    Decompose the extrinsic matrix P into R and C.
    
    Parameters
    ----------
    P : ndarray of shape (3, 4)
        Camera extrinsic matrix

    Returns
    -------
    R : ndarray of shape (3, 3)
        Camera rotation matrix
    C : ndarray of shape (3,)
        Camera center
    """
    # Extract the rotation matrix R from the extrinsic matrix P
    R = P[:, :3]
    
    # Extract the translation vector t from the extrinsic matrix P
    t = P[:, 3]
    
    # Compute the camera center C
    C = -np.linalg.inv(R) @ t
    
    return R, C
