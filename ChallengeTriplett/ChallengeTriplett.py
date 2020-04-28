import numpy as np
from skimage import io,data,exposure,color
from skimage.feature import hog, canny
import cv2

base_address = "C:\\Users\\a2tri\\source\\repos\\ComputerVision\\ChallengeTriplett\\"
set1_address = base_address + "data_batch_1"
set2_address = base_address + "data_batch_2"
set3_address = base_address + "data_batch_3"
set4_address = base_address + "data_batch_4"
set5_address = base_address + "data_batch_5"
test_set_address = base_address + "test_batch"


####################################################################################
##                                   Functions                                    ##
####################################################################################


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape(img):
    colors = np.reshape(img, (-1,1024))
    image = np.dstack([np.reshape(colors[0], (-1,32)), np.reshape(colors[1], (-1,32)), np.reshape(colors[2], (-1,32))])    
    return image

def wholeHog(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(2, 2),
                    cells_per_block=(4, 4), visualize=True, multichannel=True)
    return hog_image

def oneD(img):
    flat = np.ravel(img)
    return flat

def getHog(img):
    img = reshape(img)
    hog_image = wholeHog(img)
    flat_hog = oneD(hog_image)
    return flat_hog

def circleIntersections(dist1, dist2, dist3, p0, p1):
    #angle1 = np.arccos(((dist2**2) + (dist3**2)-(dist1**2)) / (2*dist2*dist3))
    #angle2 = np.arccos(((dist3**2) + (dist1**2)-(dist2**2)) / (2*dist3*dist1))
    #angle3 = np.arccos(((dist1**2) + (dist2**2)-(dist3**2)) / (2*dist1*dist2))
    #angles = [angle1, angle2, angle3]

    d = dist1
    r0 = dist2
    r1 = dist3

    a = ((r0**2)-(r1**2)+(d**2))/(2*d)
    h = np.sqrt((r0**2)-(a**2))

    p2 = [a*p1[0]/d, a*p1[1]/d]

    p3 = [p2[0]+(h*p1[1]/d), p2[1]+(h*p1[0]/d)]
    p4 = [p2[0]+(h*p1[1]/d), p2[1]-(h*p1[0]/d)]



    #coordinates = [[0,0], [0,dist1], [dist3*np.cos(angle3),dist3*np.sin(angle3)]]


    return angles

####################################################################################
##                                     Main                                       ##
####################################################################################

set1 = unpickle(set1_address)
set2 = unpickle(set2_address)
set3 = unpickle(set3_address)
set4 = unpickle(set4_address)
set5 = unpickle(set5_address)
testSet = unpickle(test_set_address)

for x in range(100):
    hog1 = getHog(list(set1.values())[2][x])
    hog2 = getHog(list(set1.values())[2][x+1])
    hog3 = getHog(list(set1.values())[2][x+2])

    dist1 = np.linalg.norm(hog1 - hog2)
    dist2 = np.linalg.norm(hog1 - hog3)
    dist3 = np.linalg.norm(hog3 - hog2)

    p0 = [0,0]
    p1 = [dist1,0]

    angles = circleIntersections(dist1, dist2, dist3, p0, p1)

    #io.imsave(base_address + "\\hogs\\pic" + str(x) + "_norm.jpg", img)
    #io.imsave(base_address + "\\hogs\\pic" + str(x) + "_hog.jpg", hog)

print ("Done!")