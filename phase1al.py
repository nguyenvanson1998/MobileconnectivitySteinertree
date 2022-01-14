import math

from networkx.algorithms.tree.operations import join
from utils import read_file, distance, check_inside
import numpy as np
from find_anchor import get_points_inside
import random as rd
def intersectpoint(i, j, R,dis, points):
    x1, y1 = points[i][0], points[i][1]
    x2, y2 = points[j][0], points[j][1]
    xCenter = (x1+x2)/2
    yCenter =  (y1+y2)/2
    disx =  distance(points[i], points[j])
    if  disx <= 2*R + 0.0001 and disx >= 2*R - 0.0001:
        return [[xCenter, yCenter, 0], []]


    

    xRes1 = xCenter + math.sqrt(R*R - disx*disx/4)*(y2-y1)/disx
    yRes1 = yCenter - math.sqrt(R*R - disx*disx/4)*(x2-x1)/disx
  
    xRes2 = xCenter - math.sqrt(R*R - disx*disx/4)*(y2-y1)/disx
    yRes2 = yCenter + math.sqrt(R*R - disx*disx/4)*(x2-x1)/disx
    
    return [[xRes1,yRes1,0], [xRes2,yRes2,0]]


def solve(in_path: str):
    W, H, BS, M, R, K, cars = read_file(in_path)
    points = cars

    # Save the distance of points
    dis = np.full(shape=(len(points), len(points)),fill_value= 1000000, dtype=float)
    for i in range(len(points) - 1):
        for j in range(i +1, len(points)):
            dis[i][j] = dis[j][i] = distance(points[i], points[j])


    join_point = []
    for i in range(len(points)-1):
        for j in range(i+1, len(points)):
            if dis[i][j] <= 4*R+0.0000001:
                gd1,gd2 = intersectpoint(i,j,2*R, dis,points)
                join_point.append(gd1)
                if(len(gd2) != 0):
                    join_point.append(gd2)
    
    # list_cover_of_join_point = [[] for _ in range(len(join_point)) ]
    for j in range(len(join_point)):
        for i in range(len(points)):
            if distance(join_point[j], points[i]) <= 2*R + 0.1:
                join_point[j][2] +=1


                







    # return candidate
    # candidates = []
    # for i in range(len(cars)):
    #     candidates.append(get_points_inside(i, R*2, points, dis))

    candidates = join_point
    # print(candidates)
    # boos = True
    # for point in points:
    #     bos = False
    #     for cand in candidates:
    #         if check_inside(cand, point,R):
    #             bos = True
    #             break
    #     if bos == False:
    #         boos = False
    #         break
    
    # print(boos)



    
    


    points_is_connected = []
    list_anchor = []
    # greedy choosing anchorpoint
    while(len(points_is_connected) < len(points)):
        # find the best cand in candidate_list:
        if(len(candidates) == 0):
            break
        max_connect = max(candidates, key= lambda x: x[2])[2]
        if(max_connect <= 0):
            break
        max_candidates = [i for i in range(len(candidates)) if candidates[i][2] == max_connect]
        idx = rd.choice(max_candidates)
        choosed_candidate = candidates[idx]
        list_anchor.append(choosed_candidate)
        candidates.pop(idx)

        list_inside = []
        for i in range(len(points)):
            if check_inside(choosed_candidate,points[i], R) and (i not in points_is_connected):
                list_inside.append(i)
                points_is_connected.append(i)
        
        #update the list candate
        for cand in candidates:
            for idx in list_inside:
                if check_inside(cand, points[idx], R):
                    # lst = list(cand)
                    # lst[2] = lst[2] -1
                    # cand = tuple(lst)
                    cand[2] = cand[2] - 1
    
    for i in range(len(points)):
        if i not in points_is_connected:
            idx = dis[i,:].argmin()
            cos_alpha = (points[idx][0] - points[i][0])/dis[i][idx]
            sin_alpha = (points[idx][1] - points[i][1])/dis[i][idx]
            x = points[i][0] + cos_alpha*2*R
            y = points[i][1] + sin_alpha*2*R
            list_anchor.append([x,y])


    
    return list_anchor, W, H, BS, M, R, K, cars




# cand = solve('./Testnew/12.inp')
# print(len(cand[0]))












