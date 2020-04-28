import numpy as np
from skimage import io,data,exposure,color
from skimage.feature import hog, canny
import cv2

base_address = "C:\\Users\\E\\source\\repos\\ComputerVision\\ChallengeTriplett\\"
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
    straight = list(set1.values())[2][x]
    img = reshape(straight)
    hog_image = wholeHog(img)
    flat_hog = oneD(hog_image)
    io.imsave(base_address + "\\hogs\\pic" + str(x) + "_norm.jpg", img)
    io.imsave(base_address + "\\hogs\\pic" + str(x) + "_hog.jpg", hog_image)

print ("Done!")