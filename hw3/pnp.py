import numpy as np
from tqdm import tqdm
import cv2

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation

def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3) -> 카메라 1의 1
        Set of reconstructed 3D points1
    x : ndarray of shape (n, 2) -> 카메라 2의 이미지 좌표계
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)1
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    # 이부분 꼭 바꿔야함

    success, rvec, tvec = cv2.solvePnP(X, x, np.eye(3), None)
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec.flatten()

    return R, C

def PnP_RANSAC(X, x, ransac_n_iter=10000, ransac_thr=0.05):
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
    sampling = 4
    
    for i in tqdm(range(ransac_n_iter)):
        # 6개의 인덱스 가져오기
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
    
    # TODO Your code goes here
    q = Rotation2Quaternion(R)
    p = np.hstack((C,q))
    _lamda = 5 #(1~10 중 아무 숫자나 넣으라고 나와 있음)

    front_term = 0
    back_term = 0
    for i in range(X.shape[0]):
        J = ComputePoseJacobian(p, X[i])
        f = ComputeReprojection(p, X[i])
        front_term += (J.T@J+ _lamda*np.identity(7))
        back_term += (J.T@(x[i]-f))
    
    delta_p = np.linalg.inv(front_term)@back_term
    p2 = p + delta_p
    q = p2[3:]
    #q 정규화
    q = q/np.linalg.norm(q)
    R_refined = Quaternion2Rotation(q)
    C_refined = p2[:3]

    return R_refined, C_refined

if __name__ == '__main__':
    import os
    import cv2
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from camera_pose import EstimateCameraPose

    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ],
    dtype=np.float32
    )

    track = np.load("result_npy/track.npy")
    for i in range(6):
        for j in range(i+1,6):
            track1 = track[i,:,:]
            track2 = track[j,:,:]
            
            R,C,X = EstimateCameraPose(track1, track2)
            x = track2

            #여기서 또 valid index추출해야됨
            valid_X = np.bitwise_not(X == -1)
            valid_X = np.bitwise_or(valid_X[:,0],valid_X[:,1],valid_X[:,2])
            valid_x = np.bitwise_not(x == -1)
            valid_x = np.bitwise_or(valid_x[:,0],valid_x[:,1])
            valid_index = np.bitwise_and(valid_X,valid_x)
            new_X = X[valid_index]
            new_x = x[valid_index]

            #new_x를 카메라 좌표계로
            new_x = np.hstack((new_x,np.ones((new_x.shape[0],1))))
            new_x = np.linalg.inv(K)@new_x.T
            new_x = new_x.T[:,:2]

            R,C,inlier = PnP_RANSAC(new_X, new_x)
            _X = new_X[inlier]
            _x = new_x[inlier]
            print(inlier.sum())
            print("before:",cal_reprojection_error(R,C,_X,_x).sum()/len(_x))
            R2,C2 = PnP_nl(R, C, _X, _x)
            print("after:",cal_reprojection_error(R2,C2,_X,_x).sum()/len(_x))
            #print("error : ",cal_reprojection_error(R2,C2,_X,_x).sum()/len(_x) - cal_reprojection_error(R,C,_X,_x).sum()/len(_x))

