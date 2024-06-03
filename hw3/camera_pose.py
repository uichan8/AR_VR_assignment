import numpy as np
import matplotlib.pyplot as plt

from feature import EstimateE_RANSAC

K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ])

SAVE_RESULT = True

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
    Use the linear triangulation method to triangulate the points

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
    # compute essential matrix
    valid_track1 = np.bitwise_not(track1 == -1)
    valid_track1 = np.bitwise_or(valid_track1[:,0],valid_track1[:,1])
    valid_track2 = np.bitwise_not(track2 == -1)
    valid_track2 = np.bitwise_or(valid_track2[:,0],valid_track2[:,1])
    valid_track = np.bitwise_and(valid_track1,valid_track2)
    _track1 = track1[valid_track]
    _track2 = track2[valid_track]
    valid_track = np.where(valid_track)[0]

    track1_cam_coor = np.hstack((_track1, np.ones((_track1.shape[0], 1))))
    track1_cam_coor = np.linalg.inv(K) @ track1_cam_coor.T
    track1_cam_coor = track1_cam_coor.T[:,:2]

    track2_cam_coor = np.hstack((_track2, np.ones((_track2.shape[0], 1))))
    track2_cam_coor = np.linalg.inv(K) @ track2_cam_coor.T
    track2_cam_coor = track2_cam_coor.T[:,:2]

    E, _ = EstimateE_RANSAC(track1_cam_coor, track2_cam_coor)

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
            best_valid_index = valid_index
    X_best = np.ones([track1.shape[0], 3]) * -1
    X_best[valid_track[best_valid_index]] = X[best_valid_index]

    # Visualize 3D points project Z axis
    if SAVE_RESULT:
        plt.figure()
        plt.scatter(X_best[:,1],X_best[:,2])
        plt.scatter([0],[0])
        plt.savefig("output/3D_points/3D_points.png")

    return R_best, C_best, X_best

def plot_camera(ax, position, direction, length=1.0, color='r', label='Camera'):
    # Plot the camera position
    ax.scatter(position[0], position[1], position[2], color=color, s=100, label=label)
    
    # Calculate the end point of the direction vector
    end_point = position + length * direction
    
    # Plot the direction vector
    ax.quiver(position[0], position[1], position[2],
              direction[0], direction[1], direction[2],
              color=color, length=length, arrow_length_ratio=0.1)
    

if __name__ == '__main__':
    import os
    import cv2
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    track = np.load("result_npy/track.npy")
    track1 = track[0, :, :]
    track2 = track[1, :, :]
    R, C, X = EstimateCameraPose(track1, track2)
    
    if SAVE_RESULT:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Set labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # Set axis limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # First camera position and direction
        camera1_position = np.array([0, 0, 0])
        camera1_direction = np.array([0, 0, 1])  # Assuming camera looks along Z-axis

        # Normalize the direction vector
        camera1_direction = camera1_direction / np.linalg.norm(camera1_direction)

        # Plot the first camera
        plot_camera(ax, camera1_position, camera1_direction, color='r', label='Camera 1')

        # Second camera position and direction
        camera2_position = R @ C
        camera2_direction = R @ camera1_direction

        # Normalize the direction vector
        camera2_direction = camera2_direction / np.linalg.norm(camera2_direction)

        # Plot the second camera
        plot_camera(ax, camera2_position, camera2_direction, color='b', label='Camera 2')

        # Show legend
        ax.legend()

        # Save figure
        plt.savefig("output/camera_pose/camera_poses.png")
