import math
import cv2
import numpy as np

def _getRadiusOutwards(src, axis, centre, length):
    min = -1
    max = -1
    x = centre[0]
    y = centre[1]
    pos = []
    
    for a in range(centre[0] if axis == "x" else centre[1], length-1): 

        if axis == "x":
            x = a
        else:
            y = a
            
        if any(src[y, x]) != 0: #NOT x,y!
            pos.append((x,y))
            break
            
    for a in range(centre[0]+1 if axis == "x" else centre[1]+1, 1, -1): 
        if axis == "x":
            x = a
        else:
            y = a
            
        if any(src[y, x]) != 0: #NOT x,y!
            pos.append((x,y))
            break
    
    return pos

def _getPointsOnEuclidSpiral(xMid, yMid, amount):
    pos = []
    for i in range(0, amount):
        angle = 0.5*i
        x = round(xMid + (1 + 1*angle) * math.cos(0.1*angle))
        y = round(yMid + (1 + 1*angle) * math.sin(0.1*angle))
        pos.append((x,y))
    return pos

def calculate_center(src):
    imageHeight, imageWidth, imageChannels = src.shape
    centreGuessed = [round(imageWidth/2), round(imageHeight/2)]
    print("old", centreGuessed)
    lengths = (imageWidth, imageHeight)
    MINIMAL_RADIUS_OF_CIRCLE = 20
    spiralPoints = _getPointsOnEuclidSpiral(centreGuessed[0], centreGuessed[1], 400)
    for point in spiralPoints:
        pointsToCheckX = _getRadiusOutwards(src, "x", point, lengths[0])
        pointsToCheckY = _getRadiusOutwards(src, "y", point, lengths[1])
        
        middleX = round((pointsToCheckX[0][0]+pointsToCheckX[1][0]) / 2)
        middleY = round((pointsToCheckY[0][1]+pointsToCheckY[1][1]) / 2)
        
        pointsToCheckX = _getRadiusOutwards(src, "x", [middleX, middleY], lengths[0])
        pointsToCheckY = _getRadiusOutwards(src, "y", [middleX, middleY], lengths[1])
        
        r0 = abs(middleX-pointsToCheckX[0][0])
        r1 = abs(middleX-pointsToCheckX[1][0])
        r2 = abs(middleY-pointsToCheckY[0][1])
        r3 = abs(middleY-pointsToCheckY[1][1])
        
        if r0 < MINIMAL_RADIUS_OF_CIRCLE or r1 < MINIMAL_RADIUS_OF_CIRCLE or r2 < MINIMAL_RADIUS_OF_CIRCLE  or r3 < MINIMAL_RADIUS_OF_CIRCLE:
            continue
        
        if abs(r0-r1)+abs(r2-r3) < 5:
            return [middleX, middleY]
    return centreGuessed

def alternative_center(outer_border_contour):
    # M = cv2.moments(outer_border_contour)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    #P = Polygon(outer_border_contour)
    #return P.centroid
    cX = np.mean(outer_border_contour[:,0])
    cY = np.mean(outer_border_contour[:,1])
    return (int(cX), int(cY))

def find_center(labels):
    # indices = np.unique(labels)
    # outer_border_id = indices[indices != 0].min()
    indices, counts = np.unique(labels, return_counts = True)
    indices = indices[1:]
    counts = counts[1:]
    max_count = np.argmax(counts)
    outer_border_id = indices[max_count]
    
    # Transform it into a polygon

    outer_border = np.argwhere(labels == outer_border_id).astype(np.int32)

    # Switch x and y around
    outer_border[:,[0, 1]] = outer_border[:, [1, 0]]

    # centroid = (sum(x) / len(outer_border), sum(y) / len(outer_border))
    y,x = zip(*outer_border)
    center_y, center_x = (max(x) + min(x))/2, (max(y) + min(y))/2

    return center_y, center_x