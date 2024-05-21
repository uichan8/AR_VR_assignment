import cv2
import numpy as np
from tqdm import tqdm
np.random.seed(42)



def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """
    dis_array = []

    for i in range(des1.shape[0]):
        dist = np.linalg.norm(des2 - des1[i], axis=1)
        dis_array.append(dist)
    dis_array = np.array(dis_array)
    
    argmin_des1 = np.argmin(dis_array, axis=1)
    argmin_dis2 = np.argmin(dis_array, axis=0)

    x1 = []
    x2 = []
    ind1 = []
    for i in range(des1.shape[0]):
        if argmin_dis2[argmin_des1[i]] == i:
            x1.append(loc1[i])
            x2.append(loc2[argmin_des1[i]])
            ind1.append(i)
    x1 = np.array(x1)
    x2 = np.array(x2)
    ind1 = np.array(ind1)
            
    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """

    A = np.zeros((x1.shape[0], 9))
    for i in range(x1.shape[0]):
        x1_, y1_ = x1[i]
        x2_, y2_ = x2[i]
        A[i] = np.array([x2_*x1_, x2_*y1_, x2_, y2_*x1_, y2_*y1_, y2_, x1_, y1_, 1])

    U, S, V = np.linalg.svd(A)
    E = V[-1].reshape(3, 3)
    return E



def EstimateE_RANSAC(x1, x2, ransac_n_iter = 5000, ransac_thr = 0.5):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    max_inliers = []
    best_E = None
    for _ in range(ransac_n_iter):
        idx1 = np.random.choice(x1.shape[0], 8, replace=False)
        idx2 = np.random.choice(x2.shape[0], 8, replace=False)
        temp_E = EstimateE(x1[idx1], x2[idx2])

        x1_ = np.hstack((x1, np.ones((x1.shape[0], 1))))
        x2_ = np.hstack((x2, np.ones((x2.shape[0], 1))))
        distanse_mat = (x2_.T*(temp_E@x1_.T)).sum(axis = 0)
        #distanse_mat = np.einsum('ij,ji->i', x2, E@x1.T)
        temp_inlier = np.where(np.abs(distanse_mat) < ransac_thr)[0]

        if len(temp_inlier) > len(max_inliers):
            max_inliers = temp_inlier
            best_E = temp_E
        
    E = best_E
    inlier = max_inliers
        
    return E, inlier



def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    
    SIFT_feature = []
    for i in range(Im.shape[0]):
        img = Im[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        coordinates = np.array([k.pt for k in kp])
        SIFT_feature.append((coordinates, des))

    T = max([i[0].shape[0] for i in SIFT_feature])
    #intialize the track
    track = np.ones([Im.shape[0],T,2]) * -1
    track[0,:SIFT_feature[0][0].shape[0]] = SIFT_feature[0][0]
    print("Start matching")
    for i in tqdm(range(Im.shape[0])):
        track_i = []
        for j in range(i+1,Im.shape[0]):
            if i == j:
                continue
            i_loc, i_des = SIFT_feature[i]
            j_loc, j_des = SIFT_feature[j]
            x1, x2, ind1 = MatchSIFT(i_loc, i_des, j_loc, j_des)

            #Normalize coordinate by multiplying the inverse of the intrinsic matrix
            x1 = np.hstack((x1, np.ones((x1.shape[0], 1))))
            x1 = np.linalg.inv(K)@x1.T
            x1 = x1.T[:,:2]

            x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))
            x2 = np.linalg.inv(K)@x2.T
            x2 = x2.T[:,:2]

            E, inlier = EstimateE_RANSAC(x1, x2, 10000, 0.5)
            track_i_j = np.ones_like(i_loc) * -1
            track_i_j[ind1[inlier]] = i_loc[ind1[inlier]]
            track_i.append(track_i_j)
        if i == j:
            continue
        track_i = np.array(track_i)
        valid_track_i = np.bitwise_not(track_i == -1)
        valid_track_i = valid_track_i[:,:,0] + valid_track_i[:,:,1]
        valid_track_i = np.where(valid_track_i)
        track[valid_track_i[0]+i+1,valid_track_i[1]] = track_i[valid_track_i[0],valid_track_i[1]]
    
    valid_track = np.bitwise_not(track == -1)
    valid_track = valid_track[:,:,0] + valid_track[:,:,1]
    valid_track = valid_track.sum(axis = 0) > 1
    valid_track = np.where(valid_track)[0]
    track = track[:,valid_track]

    return track

if __name__ == "__main__":
    import os
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




