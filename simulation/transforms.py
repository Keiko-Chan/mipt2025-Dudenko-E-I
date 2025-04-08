import numpy as np


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
