import math
import numpy as np
import imageio as io
import cv2

def gradients(blurred):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(float)

    xEdges = cv2.filter2D(blurred, -1, sobel_x)
    yEdges = cv2.filter2D(blurred, -1, sobel_y)

    final = np.hypot(xEdges, yEdges)        #Compute the gradient magnitude
    final = final / final.max() * 255       #The max value should be 255 
    angles = np.arctan2(xEdges, yEdges)     #Find the angle of the gradient
    angles = (angles * 180) / np.pi         #Convert to degrees

    return (final, angles)

def nms(img, angle):
    newImg = (img * 0)

    for y in range(1, len(img)-1):
        for x in range(1, len(img[0])-1):
            m = 0
            n = 0 
            #Check the adjacent pixel in the direction of the gradient magnitude
            #to determine if the current pixel should be kept
            if (0 <= angle[x,y] and angle[x,y] < 30):
                m = img[y, x+1]
                n = img[y, x-1]
            elif (150 <= angle[y,x] and angle[y,x] <= 180):
                m = img[y, x+1]
                n = img[y, x-1]
            elif (30 <= angle[y,x] and angle[y,x] < 60):
                m = img[y+1, x-1]
                n = img[y-1, x+1]
            elif (60 <= angle[y,x] and angle[y,x] < 120):
                m = img[y+1, x]
                n = img[y-1, x]
            elif (120 <= angle[y,x] and angle[y,x] < 150):
                m = img[y-1, x-1]
                n = img[y+1, x+1]

            if (img[y,x] >= m) and (img[y,x] >= n):
                newImg[y,x] = img[y,x]
            else:
                newImg[y,x] = 0
    return newImg

def dualT(img, weak):
    keep = img.max() * 0.85
    trash = img.max() * 0.20
    newImg = (img * 0)

    keep_x, keep_y = np.where(img >= keep)
    erase_x, erase_y = np.where(img < trash)
    maybe_x, maybe_y = np.where((img < keep) & (img >= trash))

    newImg[maybe_x, maybe_y] = weak
    newImg[keep_x, keep_y] = 255
    return newImg

def hyst(img, weak):
    keep = 255.0
    newImg = img
    for y in range(1, len(newImg)-1):
        for x in range(1, len(newImg[0])-1):
            if (img[x, y] == weak):
                newImg[x, y] == keep
                if ((img[x-1, y-1] == keep) or (img[x-1, y] == keep) or (img[x-1, y+1] == keep) or
                (img[x, y-1] == keep) or (img[x, y+1] == keep) or
                (img[x+1, y-1] == keep) or (img[x+1, y] == keep) or (img[x+1, y+1] == keep)):
                    newImg[x, y] == keep
                else:
                    newImg[x,y] == 0
    return newImg

def canny(imgPath, imgName):
    # Create the Gaussian Kernel
    sigma = 3
    kSize = 15 // 2
    y, x = np.mgrid[-kSize:kSize+1, -kSize:kSize+1]
    gausBase = 1 / (2 * np.pi * sigma**2)
    gaus = (gausBase * np.exp(-((x**2) + (y**2))/ (2*sigma**2)))
    weak = 255

    # Read in the image
    pathOut = imgPath + "Canny_" + imgName

    imageIn = io.imread(imgPath + imgName)
    imageOut = cv2.filter2D(imageIn, -1, gaus)
    imageOut, angles = gradients(imageOut)      #Find gradient magnitudes and thier angles
    imageOut = nms(imageOut, angles)
    
    imageOut = dualT(imageOut, weak)
    imageOut = hyst(imageOut, weak)
    imageOut = imageOut.astype(np.uint8)
    io.imwrite(pathOut, imageOut)

pathIn = "C:\\Users\\a2tri\\source\\repos\\CannyEdgesTriplett\\" 
imgName = "lena.bmp"
canny(pathIn, imgName)
pathIn = "C:\\Users\\a2tri\\source\\repos\\CannyEdgesTriplett\\"
imgName = "cameraman.tif"
canny(pathIn, imgName)
pathIn = "C:\\Users\\a2tri\\source\\repos\\CannyEdgesTriplett\\"
imgName = "fishingboat.tif"
canny(pathIn, imgName)
print ("Complete!")
