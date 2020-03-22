import numpy as np
import scipy.ndimage as nd
from skimage import io
import cv2

##################################### Functions #####################################

# Performs Gaussian Blur
# img = input image
# sigma = integer sigma value for blur
# kernel = integer dimension for kernel width, height
# returns the blurred image
def blur(img, sigma, kernel):
    # Create the Gaussian Kernel
    kSize = kernel // 2
    y, x = np.mgrid[-kSize:kSize+1, -kSize:kSize+1]
    gausBase = 1 / (2 * np.pi * sigma**2)
    gaus = (gausBase * np.exp(-((x**2) + (y**2))/ (2*sigma**2)))

    imageOut = cv2.filter2D(img, -1, gaus)
    return imageOut

# Finds image gradient magnitudes and directions with sobel kernels
# img = input image
# returns the gradient magnitude image and the gradient angles
def findGradients(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(float)

    xEdges = cv2.filter2D(img, -1, sobel_x)
    yEdges = cv2.filter2D(img, -1, sobel_y)

    final = np.hypot(xEdges, yEdges)        # Compute the gradient magnitude
    final = final / final.max() * 255       # The max value should be 255 
    angles = np.arctan2(xEdges, yEdges)     # Find the angle of the gradient
   # angles = (angles * 180) / np.pi         # Convert to degrees

    return (final, angles)

def output(img):
    pathIn = "C:\\Users\\a2tri\\source\\repos\\ComputerVision\\CircleDetectionTriplett\\Steps\\" 
    imgName = "gauges.jpg"
    pathOut = pathIn + "Step_"+ str(np.random.randint(100)) + "_" + imgName
    io.imsave(pathOut, img)


def drawCircles(img, data):
    print("Drawing Circle...")
    a = np.random.randint(0,255)
    img = cv2.circle(img, (data[1][1], data[1][0]), data[2], (a, 255, 0), 1)
    img[data[1][0]][data[1][1]] = [a,255,0]
    img[data[1][0] +1][data[1][1]] = [a,255,0]
    img[data[1][0]][data[1][1]+1] = [a,255,0]
    img[data[1][0]-1][data[1][1]] = [a,255,0]
    img[data[1][0]][data[1][1]-1] = [a,255,0]
    output(img)
    #rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #mostVotes.sort(reverse=True)
    #threshold = int(np.floor(mostVotes[0][0] * 0.9))
    #while (mostVotes[0][0] > threshold):
    #    data = mostVotes[0]

    #    del(mostVotes[0])

    return img


def rightRound(img, r, original):
    print("Creating Accumulators")
    max_r = int(min(len(img), len(img[0])) / 2) # Largest radius is half the image width
    mostVotes = []  # [vote count, (y,x), radius]
    nonzero = []    # [y,x]Votes are only cast from non-zero edge pixels
    cir_centers = []
    
    for y in range(len(img)-1):
        for x in range(len(img[0])-1):
            if (img[y][x] > 0):
                nonzero.append([y, x]) #, angles[y][x]])    # Angles might not be needed here, but it could be convenient
    
    print ("Voting")
    while (r <= max_r):
        accumulator = np.zeros((len(img), len(img[0])), dtype=np.uint64)  # Each accumulator will represent a unique radius
        centers = nonzero.copy()
        while (len(centers) > 0):


            scratch = np.zeros((len(img), len(img[0]))) # Space used to draw a circle for each edge point
            scratch = cv2.circle(scratch, (centers[0][0], centers[0][1]), r, (255,255,255), 1) # Draw a circle
            point = [] # List of all points on the circle with radius r centered at centers[0]
            point = np.where(scratch > 0)
            for i in range(0, len(point[0])):
                try:
                    accumulator[point[1][i]][point[0][i]] += 1 # Voting
                except IndexError:
                    pass
            del(centers[0])
        biggest = max(map(max, accumulator))
        index = np.unravel_index(accumulator.argmax(), accumulator.shape)
        
        print ("Greatest Votes: " + str(biggest) + ", Radius: " + str(r))
        print("85%: " + str(0.85 * accumulator[index[0]][index[1]]))
        #print ("Mean: " + str(av))
        mostVotes.insert(0, [biggest, index, r]) # most votes at (y,x) coordinates
        try:
            if (mostVotes[1][0] < (0.85 * accumulator[index[0]][index[1]])): # Large increases are circle centers
                original = drawCircles(original, mostVotes[0])
                cir_centers.insert(0, mostVotes[0][0])
                continue
            if (cir_centers[0] > (mostVotes[0][0] * 0.90)): # If the prior was a center, see if current is also
                original = drawCircles(original, mostVotes[0])
        except IndexError:
            pass
        r += 1
        

    return original

##################################### MAIN #####################################

pathIn = "C:\\Users\\a2tri\\source\\repos\\ComputerVision\\CircleDetectionTriplett\\" 
imgName = "gauges.jpg"
pathOut = pathIn + "1_" + imgName
cannyOut = pathIn + "canny_" + imgName
kernel = np.ones((5,5))

#grey = io.imread(pathIn + imgName, as_gray=True) # pixels are not uint8
img = io.imread(pathIn + imgName)
#img = blur(img, 3, 15)
#img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

canned = cv2.Canny(img, 1200, 1600, True, 5)   # Only works for color images
#gradients, angles = findGradients(canned)
#out = cv2.Canny(np.uint8(img), 100, 200, True, 5)

out = rightRound(canned, 10, img)
io.imsave(cannyOut, canned)
io.imsave(pathOut, out)

print ("Complete!")