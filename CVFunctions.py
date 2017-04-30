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
        if (ratio > 0.002) and (ratio < 0.01):
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

# accepts image with pixels 0-255
# returns image with pixels 0.0-1.0
def ones(image,ROUND=False):
    if ROUND:
        return np.round(np.divide(image.astype(float),255))
    else:
        return np.divide(image.astype(float),255)

# accepts images A and B
# displays the steps of taking the jaccard coefficient of A and B
def displayJ(A,B):
    n = np.multiply(A,B)
    u = np.subtract(np.add(A,B),n)
    display([A,B,n,u])
    
# accepts an image and size
# returns the image aspect scaled to the size
def aspectScale(image,size):
    width,height = float(image.shape[1]),float(image.shape[0])
    delta = 1
    if width > height:
        delta = float(size)/width
    if width < height:
        delta = float(size)/height
    elif width == height:
        return scale(image,size)
    dw = int(delta*width)
    dh = int(delta*height)
    img = ones(cv2.resize(image,(dw,dh)))
    width,height = img.shape[:2]
    frame = np.zeros((size,size,3))
    padTB = float(size-height)/2
    padLR = float(size-width)/2
    T,B = int(np.floor(padTB)),size-int(np.ceil(padTB))
    L,R = int(np.floor(padLR)),size-int(np.ceil(padLR))
    frame[L:R,T:B] = img
    return frame

# accepts image and resulting size
# returns image scaled to size
def scale(image,size):
    return cv2.resize(image,(size,size))

# accepts image and contours
# returns image scaled aspect to fill image frame
def aspectFill(image,contours):
    frame = image.shape
    bounds = cv2.boundingRect(contours)
    return squareCrop(image,bounds,pad=10)

# accepts image and degrees to rotate by
# returns rotated image
def rotate(image,degrees):
    rows,cols = image.shape[:2]
    mat = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, 1)
    return cv2.warpAffine(image, mat, (cols, rows))

# accepts image
# returns image processed for finding contours
def process_1(image):
    adjusted_img = image
    adjusted_img = cv2.GaussianBlur(adjusted_img, (19,11), 100)
    adjusted_img = thresh(adjusted_img,85, 255, cv2.THRESH_BINARY)
    return adjusted_img

# accepts image
# returns image processed for finding contours
def process_2(image):
    adjusted_img = image
    adjusted_img = cv2.GaussianBlur(adjusted_img, (11,11), 20)
    adjusted_img = thresh(adjusted_img,115, 255, cv2.THRESH_BINARY)
    return adjusted_img

# accepts image
# returns list of images clipped from original that may be of a license plate
def possiblePlates_1(image):
    original_img = image
    contours = findContours(process_1(original_img))
    contours = getLargeContours(contours,original_img.shape)
    possible_plate_contours = []
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        aspect = aspectRatio(w,h)
        if (aspect > 1.8) and (aspect < 3):
            possible_plate_contours.append(contours[i])
    blank_img = np.zeros(image.shape)
    if len(possible_plate_contours) != 0:
        sub_images = []
        for contour in possible_plate_contours:
            x,y,w,h = cv2.boundingRect(contour)
            sub_images.append(original_img[y:y+h,x:x+w])
        return sub_images
    return [blank_img]

# accepts image
# returns list of images clipped from original that may be of a license plate
def possiblePlates_2(image):
    original_img = image
    contours = findContours(process_2(original_img))
    contours = getLargeContours(contours,original_img.shape)
    possible_plate_contours = []
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        aspect = aspectRatio(w,h)
        if (aspect > 1.8) and (aspect < 3):
            possible_plate_contours.append(contours[i])
    blank_img = np.zeros(image.shape)
    if len(possible_plate_contours) != 0:
        sub_images = []
        for contour in possible_plate_contours:
            x,y,w,h = cv2.boundingRect(contour)
            sub_images.append(original_img[y:y+h,x:x+w])
        return sub_images
    return [blank_img]

# accepts image
# returns list of images clipped from original that may be of a license plate
def possiblePlates(image):
    return possiblePlates_1(image) + possiblePlates_2(image)

# accepts two images A,B
# returns jaccard coefficient of A,B
def J(a,b):
    A,B = ones(a),ones(b)
    n = np.multiply(A,B)
    u = np.subtract(np.add(A,B),n)
    n,u = np.sum(n),np.sum(u)
    return float(n)/float(u)


