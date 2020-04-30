import numpy as np
from skimage import io,data,exposure,color
from skimage.feature import hog, canny
from sklearn.metrics import pairwise_distances_argmin, accuracy_score
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

def kmeans(hogs):

    rand_index = np.random.randint(0, hogs.shape[0], 10)
    centers = hogs[rand_index]          # When training the centers are random

    while True:
        labels = pairwise_distances_argmin(hogs, centers)

        new_centers = np.array([hogs[labels == i].mean(0)
                                for i in range(10)])

        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

def run_test(hogs, known_centers):
    
    labels = pairwise_distances_argmin(hogs, known_centers)

    return labels


def checkAccuracy(data, labels, img_count):

    # Checking the categories the images are supposed to be in
    # against the categories they were placed in.

    label_votes = np.zeros((10, 10)) # Rows are the label that was assigned, 
                                     # Columns are a count of times that label
                                     # corresponded to a real label value
    real_totals = np.zeros((10))
    real_labels = np.zeros((10), dtype=int)
    vote_tracker = np.zeros((10))

    for i in range(img_count):
        label_votes[list(data.values())[1][i]][labels[i]] += 1
        real_totals[list(data.values())[1][i]] += 1
    
    # Now adjust the labels to correspond to the real categories
    for y in range(10):
        for x in range(10):
            vote_tracker[x] = max(label_votes[x]) # Get the max intersection for each real/measured pair
        most_votes = max(vote_tracker) # The column with the greatest vote count
        index = np.where(label_votes == most_votes)
        real_labels[index[0]] = index[1]

        label_votes[index[0][0]] = np.zeros((10)) # Once assigned clear the row
        for m in range(10):
            label_votes[m][index[1][0]] = 0

    adjusted_labels = np.zeros((labels.shape[0]), dtype=int)
    for n in range(labels.shape[0]):
        adjusted_labels[n] = real_labels[labels[n]]

    accuracy = accuracy_score(real_labels, labels)
    return accuracy

####################################################################################
##                                     Main                                       ##
####################################################################################


print ("Welcome!")
set1 = unpickle(set1_address)
set2 = unpickle(set2_address)
set3 = unpickle(set3_address)
set4 = unpickle(set4_address)
set5 = unpickle(set5_address)
testSet = unpickle(test_set_address)

img_count = 1000

hog_list = []
test_hogs = []

print ("Hogging Data...")
for x in range(img_count):
    
    hog_list.append(getHog(list(set1.values())[2][x]))
    test_hogs.append(getHog(list(testSet.values())[2][x]))


hogs = np.array(hog_list)
print ("Clustering Data...")
clusters, labels = kmeans(hogs)

hogs = np.array(test_hogs)
print ("Clustering Test Data...")
test_labels = run_test(hogs, clusters)   # Run test images with known centers

print ("Checking Accuracy...")
result = checkAccuracy(testSet, test_labels, img_count)

print ("Saving Images...")
for x in range (img_count):
    img = reshape(list(set1.values())[2][x])
    test_img = reshape(list(testSet.values())[2][x])
    io.imsave(base_address + "\\Clusters\\TrainingImages\\C" 
              + str(labels[x]) + "\\pic" + str(x) + ".jpg", img)
    io.imsave(base_address + "\\Clusters\\TestImages\\C" 
            + str(labels[x]) + "\\pic" + str(x) + ".jpg", test_img)


print ("Cleaning Up...")
for x in range(10):
    dir1 = base_address + "Clusters\\TrainingImages\\C" + str(x) + "\\"
    pics = glob.glob(dir1 + "*.jpg")
    for y in pics:
        os.remove(y)
    dir2 = base_address + "Clusters\\TestImages\\C" + str(x) + "\\"
    pics = glob.glob(dir2 + "*.jpg")
    for y in pics:
        os.remove(y)


print ("Done!")