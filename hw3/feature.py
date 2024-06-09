import cv2
import numpy as np
from scipy.spatial.distance import cdist
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
    # Compute the distance matrix between descriptors
    distance_matrix = cdist(des1, des2, 'euclidean')
    
    # Find the nearest neighbors and their distances
    nearest_neighbors_1 = np.argmin(distance_matrix, axis=1)
    nearest_distances_1 = np.min(distance_matrix, axis=1)
    
    nearest_neighbors_2 = np.argmin(distance_matrix, axis=0)
    nearest_distances_2 = np.min(distance_matrix, axis=0)
    
    # Apply the ratio test to filter out ambiguous matches
    ratio_threshold = 0.75
    valid_matches = []
    for i in range(len(nearest_neighbors_1)):
        if nearest_distances_1[i] < ratio_threshold * np.partition(distance_matrix[i], 2)[1]:
            if nearest_neighbors_2[nearest_neighbors_1[i]] == i:
                valid_matches.append((i, nearest_neighbors_1[i]))

    # Extract the matched keypoints
    x1 = np.array([loc1[i] for i, j in valid_matches])
    x2 = np.array([loc2[j] for i, j in valid_matches])
    ind1 = np.array([i for i, j in valid_matches])

    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image (normalized)
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image (normalized)

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

def EstimateE_RANSAC(x1, x2, ransac_n_iter = 200, ransac_thr = 0.001):
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
        idx = np.random.choice(x1.shape[0], 8, replace=False)
        temp_E = EstimateE(x1[idx], x2[idx])

        x1_ = np.hstack((x1, np.ones((x1.shape[0], 1))))
        x2_ = np.hstack((x2, np.ones((x2.shape[0], 1))))
        distanse_mat = np.abs(np.sum(x2_ * (temp_E @ x1_.T).T, axis=1))
        temp_inlier = np.where(distanse_mat < ransac_thr)[0]

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
    #이미지 마다 sift뽑기
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
    for i in tqdm(range(Im.shape[0])):
        track_i = np.ones([Im.shape[0],T,2]) * -1
        sift_points = SIFT_feature[i][0]
        sift_points = np.hstack((sift_points, np.ones((sift_points.shape[0], 1))))
        sift_points = np.linalg.inv(K)@sift_points.T
        sift_points = sift_points.T[:,:2]
        track_i[i,:SIFT_feature[i][0].shape[0]] = sift_points
        for j in range(i+1,Im.shape[0]):
            if i == j:
                continue
            i_loc, i_des = SIFT_feature[i]
            j_loc, j_des = SIFT_feature[j]
            x1, x2, ind1 = MatchSIFT(j_loc, j_des, i_loc, i_des)

            #Normalize coordinate by multiplying the inverse of the intrinsic matrix
            x1 = np.hstack((x1, np.ones((x1.shape[0], 1))))
            x1 = np.linalg.inv(K)@x1.T
            x1 = x1.T[:,:2] #j

            x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))
            x2 = np.linalg.inv(K)@x2.T
            x2 = x2.T[:,:2] #i

            # find inlier using essential matrix
            E, inlier = EstimateE_RANSAC(x1, x2)

            # Update track_i
            cam_coor = j_loc[ind1[inlier]].copy()
            cam_coor = np.hstack((cam_coor, np.ones((cam_coor.shape[0], 1))))
            cam_coor = np.linalg.inv(K)@cam_coor.T
            cam_coor = cam_coor.T[:,:2]
            track_i[j,ind1[inlier]] = cam_coor
            
            #Remove feature that is not matched
            matched_array = (track_i != -1)
            matched_array = matched_array[:,:,0] + matched_array[:,:,1]
            matched_index = matched_array.sum(axis = 0) > 1
            matched_array = np.bitwise_and(matched_array,matched_index)
            matched_index = np.where(matched_array)
            track[matched_index] = track_i[matched_index]
            


    #delete the feature that is not matched
    matched_array = np.bitwise_not(track == -1)
    matched_array = matched_array[:,:,0] + matched_array[:,:,1]
    matched_index = matched_array.sum(axis = 0) > 1
    matched_index = np.where(matched_index)[0]
    track = track[:,matched_index]

    return track
