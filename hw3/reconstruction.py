import numpy as np
from scipy.optimize import least_squares

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation
from utils import jacobian_BA
from utils import ComputeReprojection
from utils import cal_reprojection_error
from utils import decompose_extrinsic_matrix


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
    Valid_X = X[:,0] != -1
    Valid_track_i = track_i[:,0] != -1
    new_point = np.bitwise_and(Valid_X, Valid_track_i)

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
    _lambda = 5
    R1, C1 = decompose_extrinsic_matrix(P1)
    R2, C2 = decompose_extrinsic_matrix(P2)
    q1 = Rotation2Quaternion(R1)
    q2 = Rotation2Quaternion(R2)
    p1 = np.hstack((C1, q1))
    p2 = np.hstack((C2, q2))

    X_new = np.zeros((X.shape[0], 3))
    for j in range(X.shape[0]):
        f1 = ComputeReprojection(p1,X[j])
        f2 = ComputeReprojection(p2,X[j])
        J1 = ComputePointJacobian(X[j], p1)
        J2 = ComputePointJacobian(X[j], p2)

        E1 = np.linalg.inv(J1.T@J1 + _lambda*np.eye(3))@J1.T(x1[j] - f1)
        E2 = np.linalg.inv(J2.T@J2 + _lambda*np.eye(3))@J2.T(x2[j] - f2)
        delta_X = E1 + E2
        X_new[j] = X[j] + delta_X
        
    return X_new

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
    z = []
    # p를 만드는 구간
    for i in range(P.shape[0]):
        R,C = decompose_extrinsic_matrix(P[i])
        q = Rotation2Quaternion(R)
        p = np.hstack((C, q)).reshape(-1,1)
        z.append(p)

    #s를 만드는 구간
    s = []
    for i in range(track.shape[0]):
        s_i = FindMissingReconstruction(X, track[i])
        s.append(s_i)
    s = np.array(s) #k,J

    #b를 만드는 구간
    b = []
    for i in range(track.shape[0]):
        for j in range(track.shape[1]):
            if s[i][j]:
                b.append(track[i,j,:])
    b = np.vstack(b)
   
   #S를 만드는 구간
   


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