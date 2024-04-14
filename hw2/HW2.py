import os
import cv2
import numpy as np
from pylsd import lsd
import matplotlib.pyplot as plt

np.random.seed(42)
SAVE_IMGS = True


def DistLine2Point(line, point):
    """
    Compute the distance from a line segment to a point
    
    Parameters
    ----------
    line : ndarray of shape (4,)
        The coordinates of two points (x1, y1, x2, y2) that define the line segment
    point : ndarray of shape (2,)
        The coordinates of the point (x, y)

    Returns
    -------
    dist : float
        The distance from the line segment to the point
    """
    x1, y1, x2, y2 = line
    dist = np.abs((y2 - y1) * point[0] - (x2 - x1) * point[1] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return dist
    

def FindVP(lines, K, ransac_thr = 0.04, ransac_iter = 5000):
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
    outlier : ndarray of shape (N_l - N_i, 4)
        The set of line segments outliers
    """
    #normalize the lines
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]
    p1 = np.array([x1, y1, np.ones(x1.shape)])
    p2 = np.array([x2, y2, np.ones(x2.shape)])
    n_p1 = (np.linalg.inv(K) @ p1).T
    n_p2 = (np.linalg.inv(K) @ p2).T
    n_lines = np.hstack([n_p1[:,:2], n_p2[:,:2]])


    for i in range(ransac_iter):
        # Randomly select 2 line segments
        idx = np.random.choice(lines.shape[0], 2, replace=False)
        line1 = np.hstack([n_p1[idx[0],:2], n_p2[idx[0],:2]])
        line2 = np.hstack([n_p1[idx[1],:2], n_p2[idx[1],:2]])
        # Compute the intersection point of the two lines
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        A = np.asarray([
            [y2 - y1, x1 - x2],
            [y4 - y3, x3 - x4]
        ])
        b = np.asarray([
            (y2 - y1) * x1 - (x2 - x1) * y1,
            (y4 - y3) * x3 - (x4 - x3) * y3
        ])
        try:
            vp = np.linalg.solve(A, b)
        except:
            continue

        # Compute the inliers
        inlier = []
        inlier_max = []
        for j in range(lines.shape[0]):
            dist = DistLine2Point(n_lines[j], vp)
            if dist < ransac_thr:
                inlier.append(j)
        if len(inlier) > len(inlier_max):
            inlier_max = inlier
            vp_max = vp

        vp_max = np.array([vp_max[0], vp_max[1],1])
        vp_max = K @ vp_max
        vp_max = vp_max[:2]

    inlier = lines[inlier_max]
    outlier = np.delete(lines, inlier_max, axis=0)

    return vp_max, inlier, outlier

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
    slop = (lines[:,3] - lines[:,1]) / (lines[:,2] - lines[:,0] + 1e-7)
    angle = np.arctan(slop)
    angle = np.abs(np.degrees(angle))
    
    lines_x = lines[angle < 10]
    lines_y = lines[angle > 80]

    return lines_x, lines_y

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
    A = np.array([
        [vp_x[0]*vp_y[0]+vp_x[1]*vp_y[1],vp_x[0] + vp_y[0],vp_x[1] + vp_y[1],1],
        [vp_z[0]*vp_y[0]+vp_z[1]*vp_y[1],vp_y[0] + vp_z[0],vp_y[1] + vp_z[1],1],
        [vp_x[0]*vp_z[0]+vp_x[1]*vp_z[1],vp_x[0] + vp_z[0],vp_x[1] + vp_z[1],1]
    ])
    _,_,V = np.linalg.svd(A)
    b = V[-1,:]
    px = -b[1]/b[0]
    py = -b[2]/b[0]
    f = np.sqrt(b[3]/b[0] - (px**2 + py**2))
    K = np.array([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1]
    ])
    return K

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
    vp_x = np.array([vp_x[0], vp_x[1], 1])
    vp_y = np.array([vp_y[0], vp_y[1], 1])
    vp_z = np.array([vp_z[0], vp_z[1], 1])

    n_vp_x = np.linalg.inv(K) @ vp_x / np.linalg.norm(np.linalg.inv(K) @ vp_x)
    n_vp_y = np.linalg.inv(K) @ vp_y / np.linalg.norm(np.linalg.inv(K) @ vp_y)
    n_vp_z = np.linalg.inv(K) @ vp_z / np.linalg.norm(np.linalg.inv(K) @ vp_z)

    basis = np.array([n_vp_x, n_vp_y, n_vp_z])
    new_basis = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = basis @ np.linalg.inv(new_basis)

    H_rect = K @ R @ np.linalg.inv(K)

    return H_rect

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
    return cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))

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
    vp_z = np.array([vp_z[0], vp_z[1], 1])
    vp_x = np.array([vp_x[0], vp_x[1], 1])
    vp_y = np.array([vp_y[0], vp_y[1], 1])

    z_direction = np.linalg.inv(K) @ vp_z / np.linalg.norm(np.linalg.inv(K) @ vp_z)
    x_direction = np.linalg.inv(K) @ vp_x
    y_direction = np.linalg.inv(K) @ vp_y

    #그람 슈미트 보정
    x_direction = x_direction - np.dot(x_direction, z_direction) * z_direction
    y_direction = y_direction - np.dot(y_direction, z_direction) * z_direction

    #정규화
    x_direction = x_direction / np.linalg.norm(x_direction)
    y_direction = y_direction / np.linalg.norm(y_direction)

    #imgae 좌표계로 변환

    H = W / a

    # U
    far_center = d_far * z_direction
    U11 = far_center + H/2 * x_direction + W/2 * y_direction
    U12 = far_center + H/2 * x_direction - W/2 * y_direction
    U21 = far_center - H/2 * x_direction + W/2 * y_direction
    U22 = far_center - H/2 * x_direction - W/2 * y_direction

    # V
    near_center = d_near *  z_direction
    V11 = near_center + H/2 * x_direction + W/2 * y_direction
    V12 = near_center + H/2 * x_direction - W/2 * y_direction
    V21 = near_center - H/2 * x_direction + W/2 * y_direction
    V22 = near_center - H/2 * x_direction - W/2 * y_direction

    # #각 포인트 normalizing
    # U11 = U11 / U11[2]
    # U12 = U12 / U12[2]
    # U21 = U21 / U21[2]
    # U22 = U22 / U22[2]
    # V11 = V11 / V11[2]
    # V12 = V12 / V12[2]
    # V21 = V21 / V21[2]
    # V22 = V22 / V22[2]

    #이미지 좌표계로 변환
    # U11 = K @ U11 / U11[2]
    # U12 = K @ U12 / U12[2]
    # U21 = K @ U21 / U21[2]
    # U22 = K @ U22 / U22[2]
    # V11 = K @ V11 / V11[2]
    # V12 = K @ V12 / V12[2]
    # V21 = K @ V21 / V21[2]
    # V22 = K @ V22 / V22[2]
    
    return U11, U12, U21, U22, V11, V12, V21, V22

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
    q1 = Rotation2Quaternion(R1)
    q2 = Rotation2Quaternion(R2)
    q = q1 * (1 - w) + q2 * w
    Ri = Quaternion2Rotation(q)

    Ci = w * C1 + (1 - w) * C2
    
    return Ri, Ci

def is_left(a, b, c):
    """Return true if point c is on the left side of the line formed by points a and b"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0

def is_right(a, b, c):
    """Return true if point c is on the right side of the line formed by points a and b"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0

def is_point_in_rectangle(rect, point):
    """Check if point is inside the rectangle defined by 4 points.
       The rectangle points must be given in any order, function adjusts for orientation."""
    a, b, c, d = rect
    
    # Check the orientation of the rectangle by using the cross product of the first two edges
    if is_left(a, b, c):
        # Counter-clockwise order
        return (is_left(a, b, point) and
                is_left(b, c, point) and
                is_left(c, d, point) and
                is_left(d, a, point))
    else:
        # Clockwise order
        return (is_right(a, b, point) and
                is_right(b, c, point) and
                is_right(c, d, point) and
                is_right(d, a, point))

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
    t : ndarray of shape (3,)
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

    # vx, vy 로 그리드를 생성
    w = len(vx)
    h = len(vy)
    vx, vy = np.meshgrid(vx, vy)
    vx = vx.flatten()
    vy = vy.flatten()
    v = np.vstack([vx, vy, np.ones(vx.shape)])

    # 이미지 좌표계의 그리드를 카메라 좌표계로 변환
    p = np.linalg.inv(K) @ v

    #p22 구하기
    p22 = p21 + p12 - p11

    #p11, p12, p21, p22 를 z축이 1이 되도록 projection
    np11 = p11 / p11[2]
    np12 = p12 / p12[2]
    np21 = p21 / p21[2]
    np22 = p22 / p22[2]

    #v에서 4개의 p점을 꼭짓점으로 하는 사다리꼴 안에 있는 점들만 남기기
    new_p = []
    for i in range(p.shape[1]):
        if is_point_in_rectangle([np11[:2], np12[:2], np22[:2], np21[:2]], p[:2,i]):
            new_p.append(p[:,i])

    new_p = np.array(new_p).T
    RC = np.hstack((R, -R @ C.reshape(-1,1)))

    H = K @ RC
    Hi = H[:, :3]
    H = Hi @ np.linalg.inv(K)

    new_p = Hi @ new_p
    new_p = new_p / new_p[2,:]

    canvas = np.zeros((h, w))
    for i in range(new_p.shape[1]):
        x,y = new_p[0,i], new_p[1,i]
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        canvas[int(y), int(x)] = 1
        
        
    visibility_mask = canvas
    return H ,visibility_mask



if __name__ == '__main__':

    # Load the input image and detect the line segments
    im = cv2.imread('airport.jpg')
    im_h = im.shape[0]
    im_w = im.shape[1]
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lines = lsd(im_gray,1)

    # save the detected line segments
    if SAVE_IMGS:
        im_lines = np.copy(im)
        for i in range(lines.shape[0]):
            x1, y1, x2, y2 = lines[i,:4]
            cv2.line(im_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite('airport_lsd.jpg', im_lines)

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
    zvp, vpz_inlier, vpz_outlier = FindVP(lines[:,:4], K_apprx)

    if SAVE_IMGS:
        im_vp = np.copy(im)
        for i in range(vpz_inlier.shape[0]):
            x1, y1, x2, y2 = vpz_inlier[i]
            cv2.line(im_vp, (int(zvp[0]),int(zvp[1])), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(im_vp, (int(zvp[0]), int(zvp[1])), 5, (0, 0, 255), -1)
        cv2.imwrite('airport_vpz.jpg', im_vp)

        # save the outliers
        im_outlier = np.copy(im)
        for i in range(vpz_outlier.shape[0]):
            x1, y1, x2, y2 = vpz_outlier[i]
            cv2.line(im_outlier, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imwrite('airport_vpzoutliers.jpg', im_outlier)
    
    
	#####################################################################
    # Cluster the rest of line segments into two major directions and compute the x- and y-directional vanishing points using approximate K
	# TODO Your code goes here
    lines_x, lines_y = ClusterLines(vpz_outlier)

    #viualize the x and y linesegment
    if SAVE_IMGS:
        im_vp = np.copy(im)
        for i in range(lines_x.shape[0]):
            x1, y1, x2, y2 = lines_x[i]
            cv2.line(im_vp, (int(x1),int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite('airport_lines_x.jpg', im_vp)

        im_vp = np.copy(im)
        for i in range(lines_y.shape[0]):
            x1, y1, x2, y2 = lines_y[i]
            cv2.line(im_vp, (int(x1),int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite('airport_lines_y.jpg', im_vp)

    # Compute the x- and y-directional vanishing points
    xvp, vpx_inlier,_= FindVP(lines_x, K_apprx)
    yvp, vpy_inlier,_= FindVP(lines_y, K_apprx, 0.1)

    #visualize the x and y vanishing point
    if SAVE_IMGS:
        im_vp = np.copy(im)
        for i in range(vpx_inlier.shape[0]):
            x1, y1, x2, y2 = vpx_inlier[i]
            cv2.line(im_vp, (int(xvp[0]),int(xvp[1])), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(im_vp, (int(xvp[0]), int(xvp[1])), 5, (0, 0, 255), -1)
        cv2.imwrite('airport_vpx.jpg', im_vp)

        im_vp = np.copy(im)
        for i in range(vpy_inlier.shape[0]):
            x1, y1, x2, y2 = vpy_inlier[i]
            cv2.line(im_vp, (int(yvp[0]),int(yvp[1])), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(im_vp, (int(yvp[0]), int(yvp[1])), 5, (0, 0, 255), -1)
        cv2.imwrite('airport_vpy.jpg', im_vp)


	#####################################################################
    # Calibrate K 
    # TODO Your code goes here
    K = CalibrateCamera(xvp, yvp, zvp)

	#####################################################################
    # Compute the rectiﬁcation homography
    # TODO Your code goes here
    H = GetRectificationH(K, xvp, yvp, zvp)
    


	#####################################################################
    # Rectify the input image and vanishing points
	# TODO Your code goes here
    warp_img = ImageWarping(im, H)
    if SAVE_IMGS:
        cv2.imwrite('airport_warped.jpg', warp_img)

	#####################################################################
    # Construct 3D representation of the scene using a box model
    W = 1
    aspect_ratio = 1/2.5
    near_depth = 1.4
    far_depth = 4

	# TODO Your code goes here
    U11, U12, U21, U22, V11, V12, V21, V22 = ConstructBox(K, xvp, yvp, zvp, W, aspect_ratio, near_depth, far_depth)

    # if SAVE_IMGS:
    #     im_box = np.copy(im)
    #     cv2.line(im_box, (int(U11[0]), int(U11[1])), (int(V11[0]), int(V11[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(U12[0]), int(U12[1])), (int(V12[0]), int(V12[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(U21[0]), int(U21[1])), (int(V21[0]), int(V21[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(U22[0]), int(U22[1])), (int(V22[0]), int(V22[1])), (0, 255, 0), 2)

    #     cv2.line(im_box, (int(U11[0]), int(U11[1])), (int(U12[0]), int(U12[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(U21[0]), int(U21[1])), (int(U22[0]), int(U22[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(V11[0]), int(V11[1])), (int(V12[0]), int(V12[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(V21[0]), int(V21[1])), (int(V22[0]), int(V22[1])), (0, 255, 0), 2)

    #     cv2.line(im_box, (int(U12[0]), int(U12[1])), (int(U22[0]), int(U22[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(U11[0]), int(U11[1])), (int(U21[0]), int(U21[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(V12[0]), int(V12[1])), (int(V22[0]), int(V22[1])), (0, 255, 0), 2)
    #     cv2.line(im_box, (int(V11[0]), int(V11[1])), (int(V21[0]), int(V21[1])), (0, 255, 0), 2)
        
    #     cv2.imwrite('airport_box.jpg', im_box)
    #     im_box_warp = ImageWarping(im_box, H)
    #     cv2.imwrite('airport_box_warped.jpg', im_box_warp)
    
    
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


    interpol_num = 5
    rotations = []
    centers = []
    for i in range(len(R_list) - 1):
        for j in range(interpol_num):
            w = j / interpol_num
            Ri, Ci = InterpolateCameraPose(R_list[i], C_list[i], R_list[i+1], C_list[i+1], w)
            rotations.append(Ri)
            centers.append(Ci)

    for i in range(len(rotations)):
        canvas = np.zeros((im_h, im_w, 3))
        H, visibility_mask = GetPlaneHomography(U11, U12, U21, K, rotations[i], centers[i], np.arange(im_w), np.arange(im_h))
        plane1 = (ImageWarping(im, H) * visibility_mask[:,:,np.newaxis]).astype(np.uint8)
        canvas += plane1

        H, visibility_mask = GetPlaneHomography(V11, V12, U11, K, rotations[i], centers[i], np.arange(im_w), np.arange(im_h))
        plane2 = (ImageWarping(im, H) * visibility_mask[:,:,np.newaxis]).astype(np.uint8)
        canvas *= 1 - visibility_mask[:,:,np.newaxis]
        canvas += plane2
        canvas = canvas

        H, visibility_mask = GetPlaneHomography(U22, V22, U12, K, rotations[i], centers[i], np.arange(im_w), np.arange(im_h))
        plane3 = (ImageWarping(im, H) * visibility_mask[:,:,np.newaxis]).astype(np.uint8)
        canvas *= 1 - visibility_mask[:,:,np.newaxis]
        canvas += plane3
        canvas = canvas.astype(np.uint8)

        nU11 = H @K@ U11
        nU12 = H @K@ U12
        nU21 = H @K@ U21
        nU22 = H @K@ U22
        nV11 = H @K@ V11
        nV12 = H @K@ V12
        nV21 = H @K@ V21
        nV22 = H @K@ V22
        nV11 = nV11 / nV11[2]
        nV12 = nV12 / nV12[2]
        nV21 = nV21 / nV21[2]
        nV22 = nV22 / nV22[2]
        nU11 = nU11 / nU11[2]
        nU12 = nU12 / nU12[2]
        nU21 = nU21 / nU21[2]
        nU22 = nU22 / nU22[2]

        cv2.line(canvas, (int(nU11[0]), int(nU11[1])), (int(nV11[0]), int(nV11[1])), (0, 255, 0), 2)
        cv2.line(canvas, (int(nU12[0]), int(nU12[1])), (int(nV12[0]), int(nV12[1])), (0, 255, 0), 2)
        cv2.line(canvas, (int(nU21[0]), int(nU21[1])), (int(nV21[0]), int(nV21[1])), (0, 255, 0), 2)
        cv2.line(canvas, (int(nU22[0]), int(nU22[1])), (int(nV22[0]), int(nV22[1]),), (0, 255, 0), 2)

        cv2.line(canvas, (int(nU11[0]), int(nU11[1])), (int(nU12[0]), int(nU12[1])), (0, 255, 0), 2)
        cv2.line(canvas, (int(nU21[0]), int(nU21[1])), (int(nU22[0]), int(nU22[1])), (0, 255, 0), 2)
        cv2.line(canvas, (int(nV11[0]), int(nV11[1])), (int(nV12[0]), int(nV12[1]),), (0, 255, 0), 2)
        cv2.line(canvas, (int(nV21[0]), int(nV21[1])), (int(nV22[0]), int(nV22[1]),), (0, 255, 0), 2)

        cv2.line(canvas, (int(nU12[0]), int(nU12[1])), (int(nU22[0]), int(nU22[1])), (0, 255, 0), 2)
        cv2.line(canvas, (int(nU11[0]), int(nU11[1])), (int(nU21[0]), int(nU21[1])), (0, 255, 0), 2)
        cv2.line(canvas, (int(nV12[0]), int(nV12[1])), (int(nV22[0]), int(nV22[1]),), (0, 255, 0), 2)
        cv2.line(canvas, (int(nV11[0]), int(nV11[1])), (int(nV21[0]), int(nV21[1]),), (0, 255, 0), 2)


        if SAVE_IMGS:
            cv2.imwrite(f'results/airport_rendered_{ str(100+i)[1:]}.jpg', canvas)
        

