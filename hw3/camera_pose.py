import numpy as np

from feature import EstimateE_RANSAC
from utils import get_matching_from_track


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
    # SVD
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

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
    X = []
    for i in range(track1.shape[0]):
        x1, y1 = track1[i]
        x2, y2 = track2[i]
        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])
        _, _, Vt = np.linalg.svd(A)
        X_temp = Vt[-1]
        X_temp = X_temp / X_temp[3]
        X.append(X_temp[:3])
    X = np.array(X)

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

    X_h = np.hstack((X, np.ones((X.shape[0], 1))))

    projection1 = P1 @ X_h.T
    projection2 = P2 @ X_h.T

    projection1_z = projection1[2, :]
    projection2_z = projection2[2, :]

    valid_index = (projection1_z > 0) & (projection2_z > 0)
    
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
    _track1, _track2, valid_track = get_matching_from_track(track1, track2)
    
    E, _ = EstimateE_RANSAC(_track1, _track2)

    # Estimate four configurations of poses
    R_set, C_set = GetCameraPoseFromE(E)

    val_max = 0
    for R, C in zip(R_set, C_set):
        # Triangulate points using each configuration
        P1 = np.eye(3, 4)
        P2 = np.hstack((R, -R @ C.reshape(-1, 1)))
        X = Triangulation(P1, P2, _track1, _track2)
        # Evaluate cheirality
        valid_index = EvaluateCheirality(P1, P2, X)
        val = np.sum(valid_index)
        if val >= val_max:
            val_max = val
            R_best = R
            C_best = C
            _X_best = X
            best_valid_index = valid_index
    X_best = np.ones([track1.shape[0], 3]) * -1
    X_best[valid_track[best_valid_index]] = _X_best[best_valid_index]

    return R_best, C_best, X_best



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import cv2
    from feature import BuildFeatureTrack


    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ])
    num_images = 6
    h_im = 540
    w_im = 960

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im_file = 'im/image{:07d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    track = BuildFeatureTrack(Im, K)

    track_0 = track[0]
    track_1 = track[1]

    R, C, X = EstimateCameraPose(track_0, track_1)
    valid = X[:,0] != -1

    track_0 = track_0[valid]
    track_1 = track_1[valid]
    X = X[valid]

    #img에 track0,1 표시
    Im0 = Im[0,:,:,:]
    Im1 = Im[1,:,:,:]

    track_0_img_coor = np.hstack((track_0, np.ones((track_0.shape[0], 1))))
    track_0_img_coor = K@track_0_img_coor.T
    track_0_img_coor = track_0_img_coor[:2].T

    track_1_img_coor = np.hstack((track_1, np.ones((track_1.shape[0], 1))))
    track_1_img_coor = K@track_1_img_coor.T
    track_1_img_coor = track_1_img_coor[:2].T

    for j in range(track_0.shape[0]):
        cv2.circle(Im0, tuple(track_0_img_coor[j].astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(Im1, tuple(track_1_img_coor[j].astype(int)), 5, (0, 255, 0), -1)

    concat_image = np.concatenate((Im0, Im1), axis=1)
    plt.imshow(concat_image)
    #plt.show()
    plt.clf()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #x,y,z축 범위 설정
    ax.set_xlim(-10, 20)
    ax.set_ylim(-20, 10)
    ax.set_zlim(-0, 30)

    start1 = np.array([0, 0, 0])
    end1 = np.array([0,0,3])

    start2 = start1 + C
    end2 = R@end1 + C

    ax.quiver(start1[0], start1[1], start1[2], end1[0], end1[1], end1[2], color='blue')
    ax.quiver(start2[0], start2[1], start2[2], end2[0], end2[1], end2[2], color='red')


    vis_point = X
    vis_point = vis_point[(np.abs(vis_point) < 80).all(axis=1)]
    ax.scatter(vis_point[:,0], vis_point[:, 1], vis_point[:, 2])
    plt.show()