#!/usr/bin/env python

import numpy as np
from scipy.optimize import least_squares

def mat2quat(M):

    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1

    return q

def quat2mat(q):

    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < np.finfo(np.float).eps:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array([[ 1.0-(yY+zZ), xY-wZ, xZ+wY ], [ xY+wZ, 1.0-(xX+zZ), yZ-wX ], [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def NonLinearOptimization(param, pts1, pts2, K):
	
	rot = quat2mat([param[0], param[1], param[2], param[3]])
	col = param[4:].reshape(3, 1)
	P = np.matmul(K, rot)
	temp =  np.hstack((np.eye(3), -col))
	P = np.matmul(P, temp)

	result = 0
	for i in range(0, len(pts1)):
		
		data = pts2[i].reshape(4, 1)
		result += (pts1[i][0] - (np.matmul(P[0, :], data))/(np.matmul(P[2, :], data)))**2 + (pts1[i][1] - (np.matmul(P[1, :], data))/(np.matmul(P[2, :], data)))**2

	return result

def NonlinearPnP(imagepts, worldpts, K, pnpR, pnpC):

	quat = mat2quat(pnpR)
	initial_guess = list(quat.flatten()) + list(pnpC.flatten())
	res = least_squares(NonLinearOptimization, x0=np.array(initial_guess), args=(imagepts, worldpts, K))
	optimize_matrix = res.x

	correctedR = quat2mat([optimize_matrix[0], optimize_matrix[1], optimize_matrix[2], optimize_matrix[3]])
	correctedC = optimize_matrix[4:].reshape(3, 1)
	return correctedR, correctedC

