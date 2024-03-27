import numpy as np
import skimage

# CODE function
def GetLineFromTwoPoints(p1:np.ndarray, p2:np.ndarray):
    """
    Compute parameters for line passing two points p1 and p2

    Parameters
    ----------
    p1 : ndarray of shape (2)
        coordinate of point 1
    p2 : ndarray of shape (2)
        coordinate of point 2

    Returns
    -------
    l : ndarray of shape (3)
        parameters for line
    """
    # p1 p2 차원 추가
    points = np.array( # u v 1
        [[p1[0], p1[1], 1],
         [p2[0], p2[1], 1]]
    )
    zeros = np.array([0,0])
    # l 은 다음과 같음[a b c]1 3 3 2 
    # 둘을 곱하면 [0,0,0] 이 나와야함
    
    return l
     
# CODE function
def GetPointFromTwoLines(l1:np.ndarray, l2:np.ndarray):
    """
    Compute parameters for line passing two points p1 and p2

    Parameters
    ----------
    l1 : ndarray of shape (3)
        parameters of line 1
    l2 : ndarray of shape (3)
        parameters of line 2

    Returns
    -------
    p : ndarray of shape (3)
        coordinate for point
    """
    return p     


# CODE: load image
img = skimage.io.imread('future_hall_4f.png')

# CODE: set image size 
imw = img.shape[1]
imh = img.shape[0]


# CODE: get coordinates of 4 points of one tile in image
# [(2458, 2776), (1682, 1985), (2332, 1714), (3166, 2073)]
m11 = np.array([2458, 2776])
m12 = np.array([1682, 1985])
m21 = np.array([2332, 1714])
m22 = np.array([3166, 2073])

# CODE: set f appropriate value
f = 4000
K = np.array([[f, 0, imw/2],[0, f, imh/2],[0, 0, 1]])
K_inv = np.linalg.inv(K)

l11 = GetLineFromTwoPoints(m11,m12)
l12 = GetLineFromTwoPoints(m21,m22)
l21 = GetLineFromTwoPoints(m11,m21)
l22 = GetLineFromTwoPoints(m12,m22)
v1 = GetPointFromTwoLines(l11,l12)
v2 = GetPointFromTwoLines(l21,l22)

# CODE: DRAW lines and vanishing points overlayed on image 
# YOUR CODE HERE #
# save image with overlayed lines and points as future_hall_4f_V_[studentno].png
# YOUR CODE HERE #


r1 = K_inv@v1/np.linalg.norm(K_inv@v1)
r2 = K_inv@v2/np.linalg.norm(K_inv@v2)
r3 = np.cross(r1,r2)

R = np.column_stack((r1, r2, r3))
print("R = ",R)
print("det(R) = ",np.linalg.det(R))
print("R'R = ",R.T@R)