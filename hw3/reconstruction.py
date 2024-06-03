import numpy as np
from scipy.optimize import least_squares

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def FindMissingReconstruction(X, track_i):
    """
    Find the points that will be newly added

    Parameters
    ----------
    X : ndarray of shape (F, 3)
        3D points
    track_i : ndarray of shape (F, 2)
        2D points of the newly registered image

    Returns
    -------
    new_point : ndarray of shape (F,)
        The indicator of new points that are valid for the new image and are 
        not reconstructed yet
    """
    
    # TODO Your code goes here

    return new_point



def Triangulation_nl(X, P1, P2, x1, x2):
    """
    Refine the triangulated points

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        3D points
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    x1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    x2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X_new : ndarray of shape (n, 3)
        The set of refined 3D points
    """
    
    # TODO Your code goes here

    return X_new

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

def ComputePointJacobian(X, p):
    """
    Compute the point Jacobian

    Parameters
    ----------
    X : ndarray of shape (3,)
        3D point
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion

    Returns
    -------
    dfdX : ndarray of shape (2, 3)
        The point Jacobian
    """
    
    #forward pass
    C = p[:3]
    _C = np.eye(3,4)
    _C[:3,3] = -C

    q = p[3:]
    R = Quaternion2Rotation(q)

    _X = np.hstack((X,1))

    B = R@_C@_X
    _B = B/B[2]

    #diff
    d_B_dB = np.array([[1/B[2],0,-B[0]/B[2]**2],[0,1/B[2],-B[1]/B[2]**2],[0,0,0]])
    dB_dX =  jacobian_BA(R@_C, _X[...,np.newaxis])
    dfdX = d_B_dB @ dB_dX
    dfdX = dfdX[:2,:3]

    return dfdX

def SetupBundleAdjustment(P, X, track):
    """
    Setup bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    z : ndarray of shape (7K+3J,)
        The optimization variable that is made of all camera poses and 3D points
    b : ndarray of shape (2M,)
        The 2D points in track, where M is the number of 2D visible points
    S : ndarray of shape (2M, 7K+3J)
        The sparse indicator matrix that indicates the locations of Jacobian computation
    camera_index : ndarray of shape (M,)
        The index of camera for each measurement
    point_index : ndarray of shape (M,)
        The index of 3D point for each measurement
    """
    
    # TODO Your code goes here

    return z, b, S, camera_index, point_index
    


def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    """
    Evaluate the reprojection error

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    b : ndarray of shape (2M,)
        2D measured points
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    camera_index : ndarray of shape (M,)
        Index of camera for each measurement
    point_index : ndarray of shape (M,)
        Index of 3D point for each measurement

    Returns
    -------
    err : ndarray of shape (2M,)
        The reprojection error
    """
    
    # TODO Your code goes here

    return err



def UpdatePosePoint(z, n_cameras, n_points):
    """
    Update the poses and 3D points

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    
    # TODO Your code goes here

    return P_new, X_new



def RunBundleAdjustment(P, X, track):
    """
    Run bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    
    # TODO Your code goes here

    return P_new, X_new