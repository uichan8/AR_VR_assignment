import os
import cv2
import numpy as np
from pylsd import lsd



def FindVP(lines, K, ransac_thr, ransac_iter):
    """
    Find the vanishing point
    
    Parameters
    ----------
    lines : ndarray of shape (N_l, 4)
        Set of line segments where each row contains the coordinates of two 
        points (x1, y1, x2, y2)
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    ransac_thr : float
        Error threshold for RANSAC
    ransac_iter : int
        Number of RANSAC iterations

    Returns
    -------
    vp : ndarray of shape (2,)
        The vanishing point
    inlier : ndarray of shape (N_i,)
        The index set of line segment inliers
    """
    
    # TODO Your code goes here
    pass


def ClusterLines(lines):
    """
    Cluster lines into two sets

    Parameters
    ----------
    lines : ndarray of shape (N_l - N_i, 4)
        Set of line segments excluding the inliers from the ﬁrst vanishing 
        point detection

    Returns
    -------
    lines_x : ndarray of shape (N_x, 4)
        The set of line segments for horizontal direction
    lines_y : ndarray of shape (N_y, 4)
        The set of line segments for vertical direction
    """

    # TODO Your code goes here
    pass


def CalibrateCamera(vp_x, vp_y, vp_z):
    """
    Calibrate intrinsic parameters

    Parameters
    ----------
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction

    Returns
    -------
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    """

    # TODO Your code goes here
    pass


def GetRectificationH(K, vp_x, vp_y, vp_z):
    """
    Find a homography for rectification
    
    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction

    Returns
    -------
    H_rect : ndarray of shape (3, 3)
        The rectiﬁcation homography induced by pure rotation
    """
    
    # TODO Your code goes here
    pass


def ImageWarping(im, H):
    """
    Warp image by the homography

    Parameters
    ----------
    im : ndarray of shape (h, w, 3)
        Input image
    H : ndarray of shape (3, 3)
        Homography

    Returns
    -------
    im_warped : ndarray of shape (h, w, 3)
        The warped image
    """
    
    # TODO Your code goes here
    pass


def ConstructBox(K, vp_x, vp_y, vp_z, W, a, d_near, d_far):
    """
    Construct a 3D box to approximate the scene geometry
    
    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    vp_x : ndarray of shape (2,)
        Vanishing point in x-direction
    vp_y : ndarray of shape (2,)
        Vanishing point in y-direction
    vp_z : ndarray of shape (2,)
        Vanishing point in z-direction
    W : float
        Width of the box
    a : float
        Aspect ratio
    d_near : float
        Depth of the front plane
    d_far : float
        Depth of the back plane

    Returns
    -------
    U11, U12, U21, U22, V11, V12, V21, V22 : ndarray of shape (3,)
        The 8 corners of the box
    """
    
    # TODO Your code goes here
    pass


def InterpolateCameraPose(R1, C1, R2, C2, w):
    """
    Interpolate the camera pose
    
    Parameters
    ----------
    R1 : ndarray of shape (3, 3)
        Camera rotation matrix of camera 1
    C1 : ndarray of shape (3,)
        Camera optical center of camera 1
    R2 : ndarray of shape (3, 3)
        Camera rotation matrix of camera 2
    C2 : ndarray of shape (3,)
        Camera optical center of camera 2
    w : float
        Weight between two poses

    Returns
    -------
    Ri : ndarray of shape (3, 3)
        The interpolated camera rotation matrix
    Ci : ndarray of shape (3,)
        The interpolated camera optical center
    """
    
    # TODO Your code goes here
    pass


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
    
    # TODO Your code goes here
    pass

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

    # TODO Your code goes here
    pass


def GetPlaneHomography(p11, p12, p21, K, R, C, vx, vy):
    """
    Interpolate the camera pose
    
    Parameters
    ----------
    p11 : ndarray of shape (3,)
        Top-left corner
    p12 : ndarray of shape (3,)
        Top-right corner
    p21 : ndarray of shape (3,)
        Bottom-left corner
    K : ndarray of shape (3, 3)
        Camera intrinsic parameters
    R : ndarray of shape (3, 3)
        Camera rotation matrix
    C : ndarray of shape (3,)
        Camera optical center
    vx : ndarray of shape (h, w)
        All x coordinates in the image
    vy : ndarray of shape (h, w)
        All y coordinates in the image

    Returns
    -------
    H : ndarray of shape (3, 3)
        The homography that maps the rectiﬁed image to the canvas
    visibility_mask : ndarray of shape (h, w)
        The binary mask indicating membership to the plane constructed by p11, 
        p12, and p21
    """
    
    # TODO Your code goes here
    pass



if __name__ == '__main__':

    # Load the input image and detect the line segments
    im = cv2.imread('airport.jpg')
    im_h = im.shape[0]
    im_w = im.shape[1]
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lines = lsd(im_gray)

	# Approximate K
    f = 300
    K_apprx = np.asarray([
        [f, 0, im_w/2],
        [0, f, im_h/2],
        [0, 0, 1]
    ])

	#####################################################################
    # Compute the major z-directional vanishing point and its line segments using approximate K
	# TODO Your code goes here

	#####################################################################
    # Cluster the rest of line segments into two major directions and compute the x- and y-directional vanishing points using approximate K
	# TODO Your code goes here

	#####################################################################
    # Calibrate K 
    # TODO Your code goes here

	#####################################################################
    # Compute the rectiﬁcation homography
    # TODO Your code goes here

	#####################################################################
    # Rectify the input image and vanishing points
	# TODO Your code goes here

	#####################################################################
    # Construct 3D representation of the scene using a box model
    W = 1
    aspect_ratio = 2.5
    near_depth = 0.4
    far_depth = 4
	# TODO Your code goes here
    
	#####################################################################
    # The sequence of camera poses
    R_list = []
    C_list = []
    # Camera pose 1
    R_list.append(np.eye(3))
    C_list.append(np.zeros((3,)))
    # Camera pose 2
    R_list.append(np.asarray([
        [np.cos(np.pi/12), 0, -np.sin(np.pi/12)],
        [0, 1, 0],
        [np.sin(np.pi/12), 0, np.cos(np.pi/12)]
    ]))
    C_list.append(np.asarray([0, 0, 0.5]))
    # Camera pose 3
    R_list.append(np.asarray([
        [np.cos(np.pi/4), 0, -np.sin(np.pi/4)],
        [0, 1, 0],
        [np.sin(np.pi/4), 0, np.cos(np.pi/4)]
    ]))
    C_list.append(np.asarray([-0.1, 0, 0.4]))
    # Camera pose 4
    R_list.append(np.asarray([
        [np.cos(-np.pi/4), 0, -np.sin(-np.pi/4)],
        [0, 1, 0],
        [np.sin(-np.pi/4), 0, np.cos(-np.pi/4)]
    ]))
    C_list.append(np.asarray([0.2, 0.1, 0.6]))


	#####################################################################
    # Render images from the interpolated virtual camera poses 
	# TODO Your code goes here
