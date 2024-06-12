import numpy as np
import cv2
from tqdm import tqdm

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation
from utils import jacobian_AB
from utils import jacobian_BA
from utils import quaternion_to_rotation_matrix_jacobian
from utils import ComputeReprojection
from utils import cal_reprojection_error
from utils import decompose_extrinsic_matrix
from utils import make_projective_matrix


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    # Construct matrix A
    # num_points = X.shape[0]
    # A = []

    # for i in range(num_points):
    #     _X, _Y, _Z = X[i]
    #     _x, _y = x[i]
    #     A.append([_X, _Y, _Z, 1, 0, 0, 0, 0, -_x*_X, -_x*_Y, -_x*_Z, -_x])
    #     A.append([0, 0, 0, 0, _X, _Y, _Z, 1, -_y*_X, -_y*_Y, -_y*_Z, -_y])

    # A = np.array(A)

    # # Solve for h using SVD
    # _, _, V = np.linalg.svd(A)
    # P = V[-1].reshape(3, 4)

    # R,C = decompose_extrinsic_matrix(P)

    success, rvec, tvec = cv2.solvePnP(X, x, np.eye(3),np.zeros((5,)))
    R, _ = cv2.Rodrigues(rvec)  # 회전 벡터를 회전 행렬로 변환
    C = -np.linalg.inv(R) @ tvec  # 카메라 중심 계산
    C = C.reshape(3,)

    return R, C



def PnP_RANSAC(X, x, ransac_n_iter = 200, ransac_thr = 0.001):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    
    best_R, best_C, best_inlier = None, None, None
    best_score = 0
    sampling = 20
    
    for i in tqdm(range(ransac_n_iter)):
        index = np.random.choice(X.shape[0], sampling, replace=False)
        _X = X[index]
        _x = x[index]
    
        
        R,C = PnP(_X, _x)

        errors = cal_reprojection_error(R,C,X,x)
        inliers = errors < ransac_thr
        score = np.sum(inliers)
        
        if score > best_score:
            best_score = score
            best_R = R
            best_C = C
            best_inlier = np.array(inliers)
    
    return best_R, best_C, best_inlier



def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
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

    dB_dR = jacobian_AB(R, _C@_X[...,np.newaxis])
    dB_dRC = jacobian_AB(R@_C, _X[...,np.newaxis])
    dRC_d_C = jacobian_BA(R, _C)

    dR_dq = quaternion_to_rotation_matrix_jacobian(q)

    d_B_dq = d_B_dB @ dB_dR @ dR_dq # 3x4 -> 2x4로 바꿔야함
    df_dq = d_B_dq[:2]
    d_B_d_C = d_B_dB @ dB_dRC @ dRC_d_C # 3x12 -> 2x3로 바꿔야함

    #3,7,11 행을 때오고
    df_dC = np.array([d_B_d_C[:,3], d_B_d_C[:,7], d_B_d_C[:,11]]).T
    df_dC = -df_dC[:2]
    
    dfdp = np.hstack((df_dC, df_dq))

    return dfdp



def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    
    q = Rotation2Quaternion(R)
    p = np.hstack((C,q))
    _lamda = 5 #(1~10 중 아무 숫자나 넣으라고 나와 있음)

    front_term = 0
    back_term = 0
    for i in range(X.shape[0]):
        J = ComputePoseJacobian(p, X[i])
        f = ComputeReprojection(p, X[i])
        front_term += (J.T@J)
        back_term += (J.T@(x[i]-f))
    
    delta_p = np.linalg.inv(front_term + _lamda*np.identity(7))@back_term
    p2 = p + delta_p
    q = p2[3:]
    #q 정규화
    q = q/np.linalg.norm(q)
    R_refined = Quaternion2Rotation(q)
    C_refined = p2[:3]

    return R_refined, C_refined



if __name__ == "__main__":
    import cv2
    from feature import BuildFeatureTrack
    from camera_pose import EstimateCameraPose
    import matplotlib.pyplot as plt


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

    i = 0
    track_0 = track[i]
    track_1 = track[(i+1)%num_images]

    R, C, X = EstimateCameraPose(track_0, track_1)
    
    valid = (track_0[:,0] != -1) & (X[:,0] != -1)
    X = X[valid]
    track_0 = track_0[valid]
    R, C, inlier = PnP_RANSAC(X, track_0)

    X = X[inlier]
    track_0 = track_0[inlier]
    _R, _C = PnP_nl(R, C, X, track_0)

    plt.imshow(Im[i])
    # gt 찍기
    gt = track_0.copy()
    gt = np.hstack((gt, np.ones((gt.shape[0],1))))
    gt = K@gt.T
    gt = gt.T
    plt.scatter(gt[:,0], gt[:,1], c='r')
    
    #보정 전 찍기
    pred = X.copy()
    pred = np.hstack((pred, np.ones((pred.shape[0],1))))
    P = make_projective_matrix(R, C)
    pred = K@P@pred.T
    pred = pred.T
    plt.scatter(pred[:,0], pred[:,1], c='b')

    # #보정 후 찍기
    # pred = X.copy()
    # pred = np.hstack((pred, np.ones((pred.shape[0],1))))
    # P = make_projective_matrix(_R, _C)
    # pred = K@P@pred.T
    # pred = pred.T
    # plt.scatter(pred[:,0], pred[:,1], c='g')
    plt.show()



    

    print(cal_reprojection_error(_R,_C,X,track_0).sum() - cal_reprojection_error(R,C,X,track_0).sum())
        


    




    

