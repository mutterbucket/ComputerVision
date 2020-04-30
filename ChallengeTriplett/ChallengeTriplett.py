import numpy as np
from skimage import io,data,exposure,color
from skimage.feature import hog, canny
from sklearn.metrics import pairwise_distances_argmin
import cv2
import os
import glob

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
    return fd

def getHog(img):
    img = reshape(img)
    hog_data = wholeHog(img)
    #flat_hog = oneD(hog_image)
    return hog_data

def kmeans(hogs, known_centers):

    rand_index = np.random.randint(0, hogs.shape[0], 10)
    
    if (known_centers):
        centers = np.array(known_centers)   # When testing the centers are already known
    else:
        centers = hogs[rand_index]          # When training the centers are random

    while True:
        labels = pairwise_distances_argmin(hogs, centers)

        new_centers = np.array([hogs[labels == i].mean(0)
                                for i in range(10)])

        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


####################################################################################
##                                     Main                                       ##
####################################################################################

set1 = unpickle(set1_address)
set2 = unpickle(set2_address)
set3 = unpickle(set3_address)
set4 = unpickle(set4_address)
set5 = unpickle(set5_address)
testSet = unpickle(test_set_address)

img_count = 200

hog_list = []

print ("Hogging Data...")
for x in range(img_count):
    
    hog_list.append(getHog(list(set1.values())[2][x]))


hogs = np.array(hog_list)
print ("Clustering Data...")
clusters, labels = kmeans(hogs)
print ("Saving...")

for x in range (img_count):
    img = reshape(list(set1.values())[2][x])
    io.imsave(base_address + "\\Clusters\\C" + str(labels[x]) + "\\pic" + str(x) + ".jpg", img)
    #io.imsave(base_address + "\\hogs\\pic" + str(x) + "_hog.jpg", hog)


print ("Cleaning Up...")
for x in range(10):
    dir = base_address + "Clusters\\C" + str(x) + "\\"
    pics = glob.glob(dir + "*.jpg")
    for y in pics:
        os.remove(y)


print ("Done!")