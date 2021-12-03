from math import sqrt
import numpy as np
import math
eps = np.finfo(float).eps

def read_file(in_path: str):
    with open(in_path, "r") as f:
        W, H = [float(x) for x in f.readline().split()]
        BS = tuple([float(x) for x in f.readline().split()])
        M = int(f.readline())
        R = float(f.readline())
        K = int(f.readline())
        cars = []
        for k in range(K):
            for m in range(M):
                cars.append(tuple([float(x) for x in f.readline().split()] + [m, k]))
        return W,H, BS, M, R, K, cars

def distance(X:tuple, Y:tuple):
    return math.sqrt((X[0] - Y[0])*(X[0] - Y[0]) + (X[1] - Y[1])*(X[1] - Y[1]))

def square_distance(X:tuple, Y:tuple):
    return (X[0] - Y[0])*(X[0] - Y[0]) + (X[1] - Y[1])*(X[1] - Y[1])

def check_inside(circle: tuple, p: tuple, r_c ):
    if( square_distance(circle, p) < r_c*r_c + eps):
        return True
    else:
        return False
 