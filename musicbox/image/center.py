import math
import cv2
from shapely.geometry import Polygon
import numpy as np
import sys

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
    cX = np.mean(outer_border_contour[:,0])
    cY = np.mean(outer_border_contour[:,1])
    return (int(cX), int(cY))

def center_of_mass_filled_in(image, outer_border_contour):
    orig_image = image.copy()
    cv2.drawContours(image, [outer_border_contour], -1, (255,255,255), -1)

    greyscl = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(greyscl, 127,255,0)
    moments = cv2.moments(thresh)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    print("\nCenter of mass calculated:", (cX, cY))
    return (cX, cY)

def _translateRotation(rotation, width, height):
    #stolen: https://www.programcreek.com/python/?code=team3997%2FChickenVision%2FChickenVision-master%2FChickenVision.py
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)

def draw_ellipses(image, outer_border_contour):

    ellipse = cv2.fitEllipse(outer_border_contour)
    centerE = ellipse[0]#center of ellipse
    #rotation = ellipse[2]
    widthE = ellipse[1][0]
    heightE = ellipse[1][1]
    # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
    #rotation = translateRotation(rotation, widthE, heightE)


    cv2.ellipse(image, ellipse, (23, 184, 80), 3)
    ellipse_masks = list()

    w = int(widthE) - 1600
    h = int(heightE) - 1600
    for _ in range(0, 70):
        w = w - 12
        h = h - 12
        ellipse_mask = cv2.ellipse(image, (int(centerE[0]), int(centerE[1])), (w, h), 0.0, 0, 360, (0,0,255), 1)
        ellipse_masks.append(ellipse_masks)

    cv2.imwrite("ellipses.png", image)
    
    raise NotImplementedError("Ellipse masks to polygon...")
    return 