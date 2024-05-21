import numpy as np

from feature import EstimateE_RANSAC


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    #SVD
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U@W@V.T
    R2 = U@W.T@V.T

    C1 = U[:, 2]
    C2 = -U[:, 2]
    
    R_set = np.zeros((4, 3, 3))
    C_set = np.zeros((4, 3))
    R_set[0] = R1
    R_set[1] = R1
    R_set[2] = R2
    R_set[3] = R2

    C_set[0] = C1
    C_set[1] = C2
    C_set[2] = C1
    C_set[3] = C2

    return R_set, C_set



def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    
    # TODO Your code goes here
    val_track1 = np.bitwise_not(track1 == -1)
    val_track1 = val_track1[:,0] + val_track1[:,1]
    val_track2 = np.bitwise_not(track2 == -1)
    val_track2 = val_track2[:,0] + val_track2[:,1]
    val_track = val_track1 * val_track2
    val_track = np.where(val_track == 2)
    track1_coor = track1[val_track]
    track2_coor = track2[val_track]
    track1_coor = np.hstack((track1_coor, np.ones((track1_coor.shape[0],1))))
    track2_coor = np.hstack((track2_coor, np.ones((track2_coor.shape[0],1))))

    

    X = None
    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    
    # TODO Your code goes here

    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    # compute essential matrix
    E, _ = EstimateE_RANSAC(track1, track2)

    # Estimate four configurations of poses
    R_set, C_set = GetCameraPoseFromE(E)

    for R,C in zip(R_set,C_set):
        # Triangulate points using for each configuration
        P1 = np.eye(3,4)
        P2 = np.zeros((3,4))
        P2[:3,:3] = R
        P2[:,3] = C
        X = Triangulation(P1,P2,track1,track2)

    


    #return R, C, X

if __name__ == '__main__':
    from feature import BuildFeatureTrack
    import os
    import cv2

    img_list = os.listdir("im")
    img_stack = []
    for p in img_list:
        img = cv2.imread(os.path.join("im", p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_stack.append(img)
    img_stack = np.array(img_stack)
    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ])
    track = BuildFeatureTrack(img_stack, K)
    track1 = track[0,:,:]
    track2 = track[1,:,:]
    EstimateCameraPose(track1, track2)

