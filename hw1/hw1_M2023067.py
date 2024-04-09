import numpy as np
import skimage
from scipy.optimize import minimize

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

    # {a}x + {b}y + {c} = 0
    """
    p1 = np.array([p1[0],p1[1],1])
    p2 = np.array([p2[0],p2[1],1])
    l = np.cross(p1,p2)
    return l

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
    A = np.array([[l1[0], l1[1]],
                  [l2[0], l2[1]]])
    b = np.array([-l1[2], -l2[2]])

    p = np.linalg.solve(A, b)
    p = np.append(p,1)

    return p     

def abc2md(abc:np.ndarray):
    """
    Convert line parameters from {a}x + {b}y + {c} = 0 to y = mx + d

    Parameters
    ----------
    abc : ndarray of shape (3)
        parameters for line

    Returns
    -------
    m : float
        slope of line
    d : float
        y-intercept of line
    """
    a = abc[0]
    b = abc[1]
    c = abc[2]
    m = -a/b
    d = -c/b
    return np.array([m, d])

# CODE: load image
img = skimage.io.imread('future_hall_4f.png')
img = img[:,:,:3]
imw = img.shape[1]
imh = img.shape[0]

#이미지 오른쪽을 1000만큼 힌색 추가
white_patch = np.ones((img.shape[0], 1000, 3), dtype=np.uint8) * 255
img = np.concatenate([img, white_patch], axis=1)

# 점 추가
m11 = np.array([1385, 1681])
m12 = np.array([1927, 1539])
m21 = np.array([1233, 1526])
m22 = np.array([1690, 1437])

# 씨발
l11 = GetLineFromTwoPoints(m11,m12)
l12 = GetLineFromTwoPoints(m21,m22)
l21 = GetLineFromTwoPoints(m11,m21)
l22 = GetLineFromTwoPoints(m12,m22)

v1 = GetPointFromTwoLines(l11,l12)
v2 = GetPointFromTwoLines(l21,l22)

ll11 = abc2md(l11)
ll12 = abc2md(l12)
ll21 = abc2md(l21)
ll22 = abc2md(l22)

thickness = 5
for x in range(imh):
    try:
        y1 = int(ll11[0]*x + ll11[1])
        img[y1-thickness:y1+thickness,x-thickness:x+thickness] = np.array([255,0,0])
    except:
        pass

    try:
        y2 = int(ll12[0]*x + ll12[1])
        img[y2-thickness:y2+thickness,x-thickness:x+thickness] = np.array([0,255,0])
    except:
        pass

    try:
        y3 = int(ll21[0]*x + ll21[1])
        img[y3-thickness:y3+thickness,x-thickness:x+thickness] = np.array([0,0,255])
    except:
        pass

    try:
        y4 = int(ll22[0]*x + ll22[1])
        img[y4-thickness:y4+thickness,x-thickness:x+thickness] = np.array([255,255,0])
    except:
        pass

dot_thickness = 5
img[int(v1[1])-dot_thickness:int(v1[1])+dot_thickness,int(v1[0])-dot_thickness:int(v1[0])+dot_thickness] = np.array([255,0,255])
img[int(v2[1])-dot_thickness:int(v2[1])+dot_thickness,int(v2[0])-dot_thickness:int(v2[0])+dot_thickness] = np.array([0,255,255])

# save image with overlayed lines and points as future_hall_4f_V_[studentno].png
skimage.io.imsave('future_hall_4f_V_M2023067.png', img)


def error_function(f):
    K = np.array([[f[0], 0, imw / 2], [0, f[0], imh / 2], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    
    r1 = K_inv @ v1 / np.linalg.norm(K_inv @ v1)
    r2 = K_inv @ v2 / np.linalg.norm(K_inv @ v2)
    r3 = np.cross(r1, r2)
    
    R = np.column_stack((r1, r2, r3))
    I = np.eye(3)
    
    error = np.sum((R.T @ R - np.eye(3))**2)
    return error

# Initial guess for f
initial_guess = 1000

# Minimize the error function
result = minimize(error_function, initial_guess)

f_optimized = result.x[0] if result.success else None
print(f_optimized)


# CODE: set f appropriate value
# f = 400
# K = np.array([[f, 0, imw/2],[0, f, imh/2],[0, 0, 1]])
# K_inv = np.linalg.inv(K)

# r1 = K_inv@v1/np.linalg.norm(K_inv@v1)
# r2 = K_inv@v2/np.linalg.norm(K_inv@v2)
# r3 = np.cross(r1,r2)

# R = np.column_stack((r1, r2, r3))
# print("R = ",R)
# print("R'R = ",R.T@R)

# I = np.eye(3)
# print("det(R) = ",np.linalg.det(R))
# print(f"error : {np.linalg.norm(R.T@R - I)}")


