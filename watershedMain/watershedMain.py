import numpy as np
import imageio as io
import random as randy
import cv2
from skimage.measure import label

def forwardT(image, mask):
    timg = image
    temp = np.zeros((len(image)+2, len(image[0])+2))

    temp[0][0] = np.inf
    temp[len(temp)-1][len(temp[0])-1] = np.inf
    temp[len(temp)-1][0] = np.inf
    temp[0][len(temp[0])-1] = np.inf

    for y in range(len(image[0])):
        temp[0][y+1] = np.inf
        temp[len(temp)-1][y+1] = np.inf
        for x in range(len(image)):
            temp[x+1][y+1] = image[x][y]
            temp[x+1][0] = np.inf
            temp[x+1][len(temp[0])-1] = np.inf

    for y in range(len(image)):
        for x in range(len(image[0])):
            data = np.array([mask[0][0]+temp[y][x], 
                    mask[0][1]+temp[y][x+1],
                    mask[0][2]+temp[y][x+2],
                    mask[1][0]+temp[y+1][x],
                    mask[1][1]+temp[y+1][x+1]])
            data = np.sort(data)
            timg[y][x] = data[0]

    #print(timg)
    return timg

def backwardT(image, mask):
    timg = image
    temp = np.zeros((len(image)+2, len(image[0])+2))

    temp[0][0] = np.inf
    temp[len(temp)-1][len(temp[0])-1] = np.inf
    temp[len(temp)-1][0] = np.inf
    temp[0][len(temp[0])-1] = np.inf

    for y in range(len(image[0])):
        temp[0][y+1] = np.inf
        temp[len(temp)-1][y+1] = np.inf
        for x in range(len(image)):
            temp[x+1][y+1] = image[x][y]
            temp[x+1][0] = np.inf
            temp[x+1][len(temp[0])-1] = np.inf

    for y in range(len(image)):
        for x in range(len(image[0])):
            data = np.array([mask[0][1]+temp[y+1][x+1], 
                    mask[0][2]+temp[y+1][x+2],
                    mask[1][0]+temp[y+2][x],
                    mask[1][1]+temp[y+2][x+1],
                    mask[1][2]+temp[y+2][x+2]])
            data = np.sort(data)
            timg[y][x] = data[0]

    #print(timg)
    return timg

def transform(image, mask, reverse):
    img = forwardT(image, mask)
    img = backwardT(img, reverse)
    return img

def distance(imageIn):
    mask = np.array([[4, 3, 4], [3, 0, np.inf]])
    rev_mask = np.array([[np.inf, 0, 3], [4, 3, 4]])
    print("Finding Minima...")
    mins = imageIn.astype(float)
    minima = imageIn.astype(float)
    mins = (mins * 0) + 255
    minima = (minima * 0) + 255

    for y in range(len(mins)-1):
        for x in range(len(mins[0])-1):
            if (imageIn[y-1][x-1] < imageIn[y][x] or 
                imageIn[y-1][x] < imageIn[y][x] or
                imageIn[y-1][x+1] < imageIn[y][x] or
                imageIn[y][x-1] < imageIn[y][x] or
                imageIn[y][x+1] < imageIn[y][x] or
                imageIn[y+1][x-1] < imageIn[y][x] or
                imageIn[y+1][x] < imageIn[y][x] or
                imageIn[y+1][x+1] < imageIn[y][x]):
                    mins[y][x] = np.inf
                    minima[y][x] = 0

            if ((imageIn[y-1][x-1] == imageIn[y][x] and mins[y-1][x-1] == 0) or 
                (imageIn[y-1][x] == imageIn[y][x] and mins[y-1][x] == 0) or
                (imageIn[y-1][x+1] == imageIn[y][x] and mins[y-1][x+1] == 0) or
                (imageIn[y][x-1] == imageIn[y][x] and mins[y][x-1] == 0) or
                (imageIn[y][x+1] == imageIn[y][x] and mins[y][x+1] == 0) or
                (imageIn[y+1][x-1] == imageIn[y][x] and mins[y+1][x-1] == 0) or
                (imageIn[y+1][x] == imageIn[y][x] and mins[y+1][x] == 0) or
                (imageIn[y+1][x+1] == imageIn[y][x]) and mins[y+1][x+1] == 0):
                    mins[y][x] = np.inf
                    minima[y][x] = 0

    print("Finding Distances...")
    i = 0
    while (max(map(max, mins)) == np.inf):
        print(i)
        i += 1
        dist = transform(mins, mask, rev_mask)
        mins = dist
    mins = mins / 3 # Normalizing by 3
    return mins, minima



def assignLabels(minima):
    print("Labeling...")
    unique = label(minima)    
    return unique

def poolParty(drains, imageIn):
    print("Watershedding...")
    upstream = []

    for y in range(len(drains)-1):                 
        for x in range(len(drains[0])-1):
            upstream.append([imageIn[y][x],y,x])
                
    upstream = sorted(upstream,key=lambda x: x[0])
    #while (0 in x for x in drains):
    while len(upstream) != 0:
        x = upstream[0][2]  # Coordinates of the lowest brightness pixel
        y = upstream[0][1]
 
        # If neighbor pixel is already labeled differently, it is a watershed pixel:
        if (drains[y-1][x-1] != 0) and (drains[y-1][x-1] != drains[y][x]): 
            drains[y-1][x-1] = -1
 
        elif (drains[y-1][x] != 0) and (drains[y-1][x] != drains[y][x]):
            drains[y-1][x] = -1
    
        elif (drains[y-1][x+1] != 0) and (drains[y-1][x+1] != drains[y][x]):
            drains[y-1][x+1] = -1
   
        elif (drains[y][x-1] != 0) and (drains[y][x-1] != drains[y][x]):
            drains[y][x-1] = -1
     
        elif (drains[y][x+1] != 0) and (drains[y][x+1] != drains[y][x]):
            drains[y][x+1] = -1
    
        elif (drains[y+1][x-1] != 0) and (drains[y+1][x-1] != drains[y][x]):
            drains[y+1][x-1] = -1
 
        elif (drains[y+1][x] != 0) and (drains[y+1][x] != drains[y][x]):
            drains[y+1][x] = -1
    
        elif (drains[y+1][x+1] != 0) and (drains[y+1][x+1] != drains[y][x]):
            drains[y+1][x+1] = -1

        # If neighbor pixel is unlabeled, label it with current label
        elif (imageIn[y-1][x-1] > imageIn[y][x]) and (drains[y-1][x-1] != -1): 
            drains[y-1][x-1] = drains[y][x]
 
        elif (imageIn[y-1][x] > imageIn[y][x]) and (drains[y-1][x] != -1):
            drains[y-1][x] = drains[y][x]
    
        elif (imageIn[y-1][x+1] > imageIn[y][x]) and (drains[y-1][x+1] != -1):
            drains[y-1][x+1] = drains[y][x]
   
        elif (imageIn[y][x-1] > imageIn[y][x]) and (drains[y][x-1] != -1):
            drains[y][x-1] = drains[y][x]
     
        elif  (imageIn[y][x+1] > imageIn[y][x]) and (drains[y][x+1] != -1):
            drains[y][x+1] = drains[y][x]
    
        elif (imageIn[y+1][x-1] > imageIn[y][x]) and (drains[y+1][x-1] != -1):
            drains[y+1][x-1] = drains[y][x]
 
        elif (imageIn[y+1][x] > imageIn[y][x]) and (drains[y+1][x] != -1):
            drains[y+1][x] = drains[y][x]
    
        elif (imageIn[y+1][x+1] > imageIn[y][x]) and (drains[y+1][x+1] != -1):
            drains[y+1][x+1] = drains[y][x]

        else: # if none are greater...
            drains[y][x] = drains[y][x+1]
        del(upstream[0])
    return drains

def randColor(i):
    return [randy.randint(1,254), randy.randint(1,254), randy.randint(1,254)]

def colorTime(unique):
    print("Colorizing...")
    drains = np.zeros((len(unique), len(unique[0]), 3)) # Make an RGB image
    colors = np.zeros((max(map(max, unique)) + 2, 3))   # Make a color list
    colors = list(map(randColor, colors))
    colors[0] = [0,0,0]
    colors[1] = [255,255,255]
    drains = list(map(lambda x: list(map(lambda y: colors[y+1], x)), unique)) # Color every region based on label value
    return drains

############### Main ###############

sigma = 3
kSize = 9 // 2
y, x = np.mgrid[-kSize:kSize+1, -kSize:kSize+1]
gausBase = 1 / (2 * np.pi * sigma**2)
gaus = (gausBase * np.exp(-((x**2) + (y**2))/ (2*sigma**2)))

pathIn = "C:\\Users\\a2tri\\source\\repos\\watershedMain\\"
imgName = "moon.jpg"

kernel = np.ones((3,3), np.uint8)

imageIn = io.imread(pathIn + imgName)
imageIn = cv2.filter2D(imageIn, -1, gaus)
dist, minima = distance(imageIn)
minima = cv2.morphologyEx(minima, cv2.MORPH_CLOSE, kernel)
minima = cv2.morphologyEx(minima, cv2.MORPH_OPEN, kernel)

pathOut = pathIn + "BL3-9_ClOp3_min2" + imgName
io.imwrite(pathOut, minima)

dist, minimam = distance(minima)

labels = assignLabels(minima)
watershed = poolParty(labels, dist)
withColor = colorTime(watershed)


pathOut = pathIn + "cl5_" + imgName
io.imwrite(pathOut, withColor)
print(labels)

