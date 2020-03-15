import numpy as np
import scipy.ndimage as nd

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

mask = np.array([[4, 3, 4], [3, 0, np.inf]])
rev_mask = np.array([[np.inf, 0, 3], [4, 3, 4]])

image = np.array([[np.inf, np.inf, np.inf, np.inf],
         [np.inf, np.inf, np.inf, np.inf],
         [np.inf, np.inf, 0, np.inf],
         [np.inf, np.inf, 0, np.inf],
         [np.inf, 0, 0, np.inf],
         [np.inf, np.inf, np.inf, np.inf]])

print (image)
print()

for a in range(2):
    img = transform(image, mask, rev_mask)
    image = img
img = img / 3
print (img.astype(int))
print()


