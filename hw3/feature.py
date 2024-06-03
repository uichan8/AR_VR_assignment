import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(42)

SAVE_RESULT = True
MODE = None # cv2 or None

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
    if MODE == "cv2":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        match_result = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                match_result.append(m)

        x1 = np.array([loc1[m.queryIdx] for m in match_result])
        x2 = np.array([loc2[m.trainIdx] for m in match_result])
        ind1 = np.array([m.queryIdx for m in match_result])

    else:
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

def EstimateE_RANSAC(x1, x2, ransac_n_iter = 10000, ransac_thr = 0.001):
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
        #distanse_mat = np.einsum('ij,ji->i', x2, E@x1.T)
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
    
    SIFT_feature = []
    for i in range(Im.shape[0]):
        img = Im[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        coordinates = np.array([k.pt for k in kp])
        SIFT_feature.append((coordinates, des))
        # visulize sift points
        if SAVE_RESULT:
            result_img = cv2.drawKeypoints(gray, kp, img.copy())
            cv2.imwrite(f'output/SIFT/sift_{i}.png', result_img)
            plt.clf()

    T = max([i[0].shape[0] for i in SIFT_feature])

    #intialize the track
    track = np.ones([Im.shape[0],T,2]) * -1
    print("Start matching")
    for i in tqdm(range(Im.shape[0])):
        track_i = np.ones([Im.shape[0],T,2]) * -1
        track_i[i,:SIFT_feature[i][0].shape[0]] = SIFT_feature[i][0]
        for j in range(i+1,Im.shape[0]):
            if i == j:
                continue
            i_loc, i_des = SIFT_feature[i]
            j_loc, j_des = SIFT_feature[j]
            x1, x2, ind1 = MatchSIFT(j_loc, j_des, i_loc, i_des)

            #visulize matching point
            if SAVE_RESULT:
                img1 = Im[i]
                img2 = Im[j]
                concat_img = np.hstack((img1, img2))
                plt.imshow(concat_img)
                count = 0
                for p1,p2 in zip(x2, x1):
                    count += 1
                    if count % 50 == 0:
                        plt.plot([p1[0], p2[0]+img1.shape[1]], [p1[1], p2[1]], 'r')
                plt.savefig(f"output/match_SIFT/match_{i}_{j}.png")
                plt.clf()

            #Normalize coordinate by multiplying the inverse of the intrinsic matrix
            x1_img_coor = x1.copy()
            x2_img_coor = x2.copy()
            
            x1 = np.hstack((x1, np.ones((x1.shape[0], 1))))
            x1 = np.linalg.inv(K)@x1.T
            x1 = x1.T[:,:2] #j

            x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))
            x2 = np.linalg.inv(K)@x2.T
            x2 = x2.T[:,:2] #i

            # find inlier using essential matrix
            if MODE == "cv2":
                E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                mask = mask.ravel().astype(bool)
                inlier = np.where(mask)[0]
            else:
                E, inlier = EstimateE_RANSAC(x1, x2, 10000, 0.001)

            # visulize epipolar line
            if SAVE_RESULT:
                F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
                img1 = Im[i]
                img2 = Im[j]
                lines1 = compute_epilines(x1_img_coor[inlier], 2, F)
                lines2 = compute_epilines(x2_img_coor[inlier], 1, F)
                img1, img2 = draw_epilines(img1, img2, lines2, x2_img_coor[inlier], x1_img_coor[inlier])
                img2, img1 = draw_epilines(img2, img1, lines1, x1_img_coor[inlier], x2_img_coor[inlier])
                plt.imshow(img1)
                plt.savefig(f"output/epipolar_line/epipolar_{i}_{j}_1.png")
                plt.clf()
                plt.imshow(img2)
                plt.savefig(f"output/epipolar_line/epipolar_{i}_{j}_2.png")
                plt.clf()

            # Update track_i
            track_i[j,ind1[inlier]] = j_loc[ind1[inlier]] # 만약 카메라 좌표계를 사용해야 한다면 이 부분을 수정해야 함

            #visulize inlier
            if SAVE_RESULT:
                x1_img_coor = x1_img_coor[inlier]
                x2_img_coor = x2_img_coor[inlier]
                img1 = Im[i]
                img2 = Im[j]
                concat_img = np.hstack((img1, img2))
                plt.imshow(concat_img)
                count = 0
                for p1,p2 in zip(x2_img_coor, x1_img_coor):
                    count += 1
                    if count % 50 == 0:
                        plt.plot([p1[0], p2[0]+img1.shape[1]], [p1[1], p2[1]], 'r')
                plt.savefig(f"output/match_without_outlier/inlier_{i}_{j}.png")
                plt.clf()
            
            #Remove feature that is not matched
            matched_array = np.bitwise_not(track_i == -1)
            matched_array = matched_array[:,:,0] + matched_array[:,:,1]
            matched_index = matched_array.sum(axis = 0) > 1
            matched_array = np.bitwise_and(matched_array,matched_index)
            matched_index = np.where(matched_array)

            #sum up the track
            # track_i_camarea_coor = track_i[matched_index]
            # track_i_camarea_coor = np.hstack((track_i_camarea_coor, np.ones((track_i_camarea_coor.shape[0], 1))))
            # track_i_camarea_coor = np.linalg.inv(K)@track_i_camarea_coor.T
            # track_i_camarea_coor = track_i_camarea_coor.T[:,:2]
            # track[matched_index]= track_i_camarea_coor
            track[matched_index] = track_i[matched_index]

    #delete the feature that is not matched
    matched_array = np.bitwise_not(track == -1)
    matched_array = matched_array[:,:,0] + matched_array[:,:,1]
    matched_index = matched_array.sum(axis = 0) > 1
    matched_index = np.where(matched_index)[0]
    track = track[:,matched_index]

    return track
# 에피포라 라인 계산
def compute_epilines(pts, which_image, E):
    pts = np.array(pts, dtype=np.float32)
    lines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), which_image, E)
    lines = lines.reshape(-1, 3)
    return lines

# 에피포라 라인 그리기 함수
def draw_epilines(img1, img2, lines, pts1, pts2):
    r, w, c = img1.shape
    img1_color = img1.copy()
    img2_color = img2.copy()
    
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0]*w) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(pt1.astype(int)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(pt2.astype(int)), 5, color, -1)
    return img1_color, img2_color


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

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
    print(track.shape)
    np.save("result_npy/track.npy", track)

    

        





