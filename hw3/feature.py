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
    temp_track = []
    for i in tqdm(range(Im.shape[0]-1)):
        track_i = np.ones([Im.shape[0],T,2]) * -1
        for j in range(i+1,Im.shape[0]):
            i_loc, i_des = SIFT_feature[i]
            j_loc, j_des = SIFT_feature[j]
            x1, x2, ind1 = MatchSIFT(i_loc, i_des, j_loc, j_des)

            #Normalize coordinate by multiplying the inverse of the intrinsic matrix
            x1 = np.hstack((x1, np.ones((x1.shape[0], 1))))
            x1 = np.linalg.inv(K)@x1.T
            x1 = x1.T[:,:2] #i

            x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))
            x2 = np.linalg.inv(K)@x2.T
            x2 = x2.T[:,:2] #j

            # find inlier using essential matrix
            E, inlier = EstimateE_RANSAC(x1, x2)

            # Update track_i
            x1 = x1[inlier]
            x2 = x2[inlier]
            track_i[i,ind1[inlier]] = x1
            track_i[j,ind1[inlier]] = x2

        #Remove feature that is not matched
        valid_track_index = (track_i[:,:,0]!=-1).sum(axis = 0)!=0
        track_i = track_i[:,valid_track_index]
        temp_track.append(track_i)

    #aggregate_track
    track = np.zeros([Im.shape[0],0,2])
    for i in range(len(temp_track)):
        target_track = temp_track[i]
        valid_target_index = np.array([target_track[i][j] not in track[i] for j in range(target_track[i].shape[0])])
        track = np.concatenate((track, target_track[:,valid_target_index]), axis=1)

    return track

#visualize epilines
def draw_epilines(img1, img2, pts1, pts2, K):
    # Find the fundamental matrix
    E, inlier = EstimateE_RANSAC(pts1, pts2)
    
    # Select only inlier points
    pts1 = pts1[inlier]
    pts2 = pts2[inlier]

    #to img coordinate
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts1 = K@pts1.T
    pts1 = pts1.T[:,:2]

    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    pts2 = K@pts2.T
    pts2 = pts2.T[:,:2]

    # E 2 F
    F = np.linalg.inv(K).T@E@np.linalg.inv(K)
    
    # Find epilines corresponding to points in the right image (second image) and drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    
    img1_epilines = draw_lines(img1, lines1, pts1)
    
    # Find epilines corresponding to points in the left image (first image) and drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    img2_epilines = draw_lines(img2, lines2, pts2)
    
    return img1_epilines, img2_epilines

def draw_lines(img, lines, pts):
    r, c = img.shape[:2]
    img_lines = img.copy()
    for r, pt in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img_lines = cv2.line(img_lines, (x0, y0), (x1, y1), color, 1)
        img_lines = cv2.circle(img_lines, tuple(pt.astype(int)), 5, color, -1)
    return img_lines

if __name__ == "__main__":
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
    
    valid = np.bitwise_and(track_0[:,0] != -1,track_1[:,0] != -1)
    track_0 = track_0[valid]
    track_1 = track_1[valid]

    img1_epilines, img2_epilines = draw_epilines(Im[0], Im[1], track_0, track_1, K)

    concat_image = np.concatenate([img1_epilines, img2_epilines], axis=1)
    concat_image_rgb = cv2.cvtColor(concat_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Concatenated Image', concat_image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

