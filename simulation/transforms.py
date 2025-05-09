import numpy as np
import cv2


def warp_point(x: int, y: int, M) -> tuple[int, int]:
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    return (int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d), 
            int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d))


def warp_quad(quad, M):
    p1 = warp_point(quad[0][0], quad[0][1], M)
    p2 = warp_point(quad[1][0], quad[1][1], M)
    p3 = warp_point(quad[2][0], quad[2][1], M)
    p4 = warp_point(quad[3][0], quad[3][1], M)

    return [p1, p2, p3, p4]
    

def rotate_quad(quad, R_inv):
    points = np.array(quad)
    
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    result = R_inv.dot(points_ones.T).T
       
    return result
    

def create_2d_gaussian(size, sigma=1.0):
    x = np.linspace(-size//2, size//2, size)
    y = np.linspace(-size//2, size//2, size)
    x, y = np.meshgrid(x, y)
    distance = np.sqrt(x**2 + y**2)
    gaussian = np.exp(-(distance**2) / (2 * sigma**2))
    return gaussian


def smooth_mask(mask_inp, kernel_size):
    mask = np.array(mask_inp) / 255
    result_mask = mask.copy()
    
    def funct(area):
        if area[kernel_size // 2, kernel_size // 2] == 0:
            return 0
        
        v_sum = np.sum(area)
        weight = v_sum / kernel_size/ kernel_size
        
        return weight
     
    pad = kernel_size // 2
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    one_indices = np.argwhere(padded == 1)

    for ind in one_indices:
        if (ind[0] - pad < 0 or ind[1] - pad < 0 or 
            ind[0] - pad >= result_mask.shape[0] or ind[1] - pad >= result_mask.shape[1]):
            continue
        area = padded[ind[0]-pad:ind[0]+pad+1, ind[1]-pad:ind[1]+pad+1]
        result_mask[ind[0] - pad, ind[1] - pad] = funct(area)
     
    result = np.stack([result_mask] * 3, axis=-1) 
    return result


def pyramid_blending(background_inp, warped_inp, mask_inp, levels=5):
    background = np.array(background_inp)
    mask = np.array(mask_inp) / 255
    warped = np.array(warped_inp)

    A = warped
    B = background
    
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(gpA[i])
        gpA.append(G)
     
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(gpB[i])
        gpB.append(G)
     
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize = size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)
     
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize = size)
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)
     
    LS = []
    for la,lb in zip(lpA,lpB):
        h_lap, w_lap = la.shape[:2]
        mask_resized = cv2.resize(mask, (w_lap, h_lap), interpolation=cv2.INTER_LINEAR)
        mask_up = np.stack([mask_resized] * 3, axis=-1) 
        ls = lb * (1 - mask_up) + la * mask_up
        LS.append(ls)
     
    ls_ = LS[0]
    for i in range(1,6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize = size)
        ls_ = cv2.add(ls_, LS[i])
    
    blended_img = np.clip(ls_, 0, 255).astype(np.uint8)
    
    return blended_img
