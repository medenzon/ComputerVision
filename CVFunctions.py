import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import os

#%matplotlib inline

# accepts image filename as string
# reads and returns image
def load(filename):
    return cv2.imread(filename)


def aspectRatio(w,h):
    return float(w)/float(h)

# accepts an image
# draws image
def draw(image,title='',axis=True,gray=True,cmap=''):
    plt.figure()
    plt.title(title)
    if gray or cmap != '':
        if gray:
            plt.imshow(image,cmap='gray')
        else:
            plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
        plt.show()
    if not axis:
        plt.axis('off')

# accepts a list of images
# displays images horizontally
def display(images,gray=True):
    figure = plt.figure()
    for i in xrange(len(images)):
        figure.add_subplot(1,len(images),i+1)
        if gray:
            plt.imshow(images[i],cmap='gray')
        else:
            plt.imshow(images[i])
        plt.axis( 'off' )
        
# accepts an image
# returns opencv contours
def findContours(image):
    img = image
    #img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

# accepts opencv contours
# converts to image and draws contours
def drawContours(contours,mask,stroke=10):
    draw(imageFromContours(contours,mask,stroke))
    
# accepts list of opencv contours and image shape
# returns a list of larger contours
def getLargeContours(contours,shape):
    largeContours = []
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        ratio = cv2.contourArea(contours[i])/(shape[0]*shape[1])
        if (ratio > 0.002) and (ratio < 0.1):
            largeContours.append(contours[i])
    return largeContours
    
# accepts opencv contours
# returns contours as opencv image
def imageFromContours(contours,img,stroke=5):
    return cv2.drawContours(np.zeros(img.shape,np.uint8),contours, -1,(255,0,0),stroke)

# accepts opencv contours
# returns image from masked countours
def maskFromContours(contours,img):
    return cv2.drawContours(np.zeros(img.shape,np.uint8),contours,-1,(255,255,255),cv2.FILLED)

# accepts image and threshold parameters
# returns threshold image
def thresh(image,threshVal,maxVal,threshType):
    ret,threshImg = cv2.threshold(image,threshVal,maxVal,threshType)
    return threshImg

# accepts contours and image shape
# returns contours as image scaled and centered in 200x200 frame
def formatContours(contours,image,pad=20):
    frame = image.shape
    bounds = cv2.boundingRect(contours)
    x,y,w,h = bounds
    stroke = 32
    img = aspectScale(image[y:y+h,x:x+w],200)
    return img

# accepts image
# returns image contours as image scaled and centered in 200x200 frame
def formatImage(image,stroke=3):
    contours = findContours(image)
    img = imageFromContours(contours[0],image.shape,stroke)
    img = aspectFill(image,contours[0])
    img = scale(image,200)
    return img

# accepts image with pixels 0-255
# returns image with pixels 0.0-1.0
def ones(image):
    return np.round(np.divide(image.astype(float),255))

# accepts images A and B
# displays the steps of taking the jaccard coefficient of A and B
def displayJ(A,B):
    n = np.multiply(A,B)
    u = np.subtract(np.add(A,B),n)
    display([A,B,n,u])
