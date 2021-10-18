"""
    Some codes are from somewhere, I couldn't remember but finding the original version that i got. 
    And Some are mine.
    These all are for test and understand how they use and why
"""







import cv2
from matplotlib import pyplot as plt


"""
    https://stackoverflow.com/questions/60759031/computer-vision-creating-mask-of-hand-using-opencv-and-python
    erodw-dilate
"""
def read_image():
    img = cv2.imread("pic.jpg", cv2.IMREAD_ANYCOLOR) # If set, the image is read in any possible color format.
    img_2 = cv2.imread("pic.jpg", cv2.IMREAD_ANYDEPTH) # If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
    img_3 = cv2.imread("pic.jpg", cv2.IMREAD_COLOR) # 1 : If set, always convert image to the 3 channel BGR color image.
    img_4 = cv2.imread("pic.jpg", cv2.IMREAD_GRAYSCALE) # 0 : If set, always convert image to the single channel grayscale image (codec internal conversion).
    img_5 = cv2.imread("pic.jpg", cv2.IMREAD_IGNORE_ORIENTATION) # If set, do not rotate the image according to EXIF's orientation flag.
    img_6 = cv2.imread("pic.jpg", cv2.IMREAD_LOAD_GDAL) # If set, use the gdal driver for loading the image.
    img_7 = cv2.imread("pic.jpg", cv2.IMREAD_REDUCED_COLOR_2) # If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2. 
    img_8 = cv2.imread("pic.jpg", cv2.IMREAD_REDUCED_COLOR_4) # If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
    img_9 = cv2.imread("pic.jpg", cv2.IMREAD_REDUCED_COLOR_8) # If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
    img_10 = cv2.imread("pic.jpg", cv2.IMREAD_UNCHANGED) # -1 : If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
    img_11 = cv2.imread("pic.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_2) # If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
    img_12 = cv2.imread("pic.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_4) # If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
    img_13 = cv2.imread("pic.jpg", cv2.IMREAD_REDUCED_GRAYSCALE_8) # If set, always convert image to the single channel grayscale image and the image size reduced 1/8.

    # cv2.imshow("name to show", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows
    
    # https://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/ 
    plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122),plt.imshow(cv2.cvtColor(img_11, cv2.COLOR_BGR2RGB))
    plt.show()

# read_image()

def drawing_function():
    return()


# remove straymarks
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('1.png')
wimg = img[:, :, 0]
ret,thresh = cv2.threshold(wimg,100,255,cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((13,13), np.uint8)
erosion = cv2.erode(closing, kernel, iterations = 1)
mask = cv2.bitwise_or(erosion, thresh)
white = np.ones(img.shape,np.uint8)*255
white[:, :, 0] = mask
white[:, :, 1] = mask
white[:, :, 2] = mask
result = cv2.bitwise_or(img, white) # <class 'numpy.ndarray'>
mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
mask = cv2.bitwise_not(mask)
output = cv2.inpaint(img, mask,3, cv2.INPAINT_TELEA)

# plt.subplot(121),plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
# plt.subplot(122),plt.imshow(output)
# plt.show()
# img_filtered = cv2.filter2D(result, cv2.CV_8U, gb_kernel.transpose())

# print(type(result))
# dst = cv2.inpaint(img, result, 3, cv2.INPAINT_TELEA)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
# lower_blue = np.array([100,50,50]) 
# upper_blue = np.array([150,255,255]) 
# kernel = np.ones((5,5),np.uint8)
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# mask = cv2.dilate(mask,kernel,iterations = 4)
# dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


cv2.imwrite("2.png",output)

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from numpy.lib.type_check import imag
from matplotlib import pyplot as plt

def seal_1():
    img = cv2.imread("veidz.jpg")

    alpha = 2.0
    beta = -160

    new = alpha * img + beta
    new = np.clip(new, 0, 255).astype(np.uint8)

    cv2.imwrite("cleaned.png", new)

def seal_2():
    image = cv2.imread('1.png')
    image = Image.fromarray(image)
    image_contrast = ImageEnhance.Contrast(image).enhance(1.5)

    img_hsv = cv2.cvtColor(np.array(image_contrast)[:, :, ::-1],
                            cv2.COLOR_BGR2HSV)

    red_lower = np.array([110, 50, 50], np.uint8)
    red_upper = np.array([200, 255, 255], np.uint8)
    red_mask = cv2.inRange(img_hsv, red_lower, red_upper)

    kernal = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernal)
    img = np.array(image)
    dst = cv2.inpaint(img, red_mask, 0 , cv2.INPAINT_TELEA)
    # cv2.imshow("xx.png",dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.subplot(121),plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # plt.subplot(122),plt.imshow(cv2.cvtColor(cv2.bitwise_not(threshInv), cv2.COLOR_BGR2RGB))
    plt.show()

def seal_3():

    img = cv2.imread('messi_2.jpg')
    mask = cv2.imread('mask2.png',0)
    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def seal_4():

    inp_img = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)
    th,inp_img_thresh = cv2.threshold(255-inp_img,220,255,cv2.THRESH_BINARY)
    dilate = cv2.dilate(inp_img_thresh,np.ones((5,5),np.uint8))
    canny = cv2.Canny(dilate,0,255)
    contours,_ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    test_img = inp_img.copy()
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        #print(x,y,w,h,test_img[y+h//2,x-w])
        test_img[y+3:y-2+h,x+3:x+w] = 240 #test_img[y+h//2,x-w]

    cv2.imwrite("stamp_removed.jpg",test_img)
    cv2.imshow("input image",inp_img)
    cv2.imshow("threshold",inp_img_thresh)
    cv2.imshow("output image",test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def seal_5():
    originalImage = cv2.imread('1.png')
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    thresh, blackAndWhiteImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Black white image', blackAndWhiteImage)
    # cv2.imshow('Original image',originalImage)
    # cv2.imshow('Gray image', grayImage)
    plt.subplot(121),plt.imshow(cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_BGR2RGB))
    plt.subplot(122),plt.imshow(cv2.cvtColor(cv2.bitwise_not(grayImage), cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imwrite("stamp_removed.jpg",img)
seal_5()


import re
import numpy as np
import cv2 
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\PeerasakChatsermsak\AppData\Local\Tesseract-OCR\tesseract.exe"

"""
OpenCV provides four variations of this technique.

- cv2.fastNlMeansDenoising() - works with a single grayscale images
- cv2.fastNlMeansDenoisingColored() - works with a color image.
- cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
- cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
"""

def noise_1():
    # cv.fastNlMeansDenoisingColored()
    """ 
    As mentioned above it is used to remove noise from color images. (Noise is expected to be gaussian).
    """
    img = cv2.imread('1.png')
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    print("IMG Type ", type(img), " DST type ", type(dst))
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.show()

def noise_2():
    # load the image and display it
    image = cv2.imread("1.png")
    # cv2.imshow("Image", image)
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # apply Otsu's automatic thresholding
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.imshow("Otsu Thresholding", threshInv)
    # cv2.waitKey(0)
    # instead of manually specifying the threshold value, we can use
    # adaptive thresholding to examine neighborhoods of pixels and
    # adaptively threshold each neighborhood
    thresh_mean = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # cv2.imshow("Mean Adaptive Thresholding", thresh)
    # cv2.waitKey(0)

    # perform adaptive thresholding again, this time using a Gaussian
    # weighting versus a simple mean to compute our local threshold
    # value
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    # cv2.imshow("Gaussian Adaptive Thresholding", thresh)
    # cv2.waitKey(0)
    # print(type(threshInv))

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    # x = cv2.erode(threshInv, kernel, iterations = 1)
    
    plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(122),plt.imshow(cv2.cvtColor(cv2.bitwise_not(threshInv), cv2.COLOR_BGR2RGB))
    plt.show()

# noise_2()
# def unsharp_mask(img, blur_size = (9,9), imgWeight = 1.5, gaussianWeight = -0.5):
#     gaussian = cv2.GaussianBlur(img, (5,5), 0)
#     return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)

# img_file = '1.png'
# img = cv2.imread(img_file, cv2.IMREAD_COLOR)
# img = cv2.blur(img, (5, 5))
# img = unsharp_mask(img)
# img = unsharp_mask(img)
# img = unsharp_mask(img)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)

# thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# contours, heirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = sorted(contours, key = cv2.contourArea, reverse = True)
# #for cnt in cnts:
# canvas_for_contours = thresh.copy()
# cv2.drawContours(thresh, cnts[:-1], 0, (0,255,0), 3)
# cv2.drawContours(canvas_for_contours, contours, 0, (0,255,0), 3)
# # cv2.imshow('Result', canvas_for_contours - thresh)
# cv2.imwrite("result.jpg", canvas_for_contours - thresh)
# # cv2.waitKey(0)

# plt.subplot(121),plt.imshow(canvas_for_contours - thresh)
# plt.subplot(122),plt.imshow(cv2.imread("result.jpg"))
# plt.show()
def noise_3():
    img = cv2.imread('1.png', 0) 
    ret, bw = cv2.threshold(img, 128,255,cv2.THRESH_BINARY_INV) #use to be binary before remove noise
    connectivity = 1
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 3 # threshhold value for small noisy components
    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        # print("2")
        if sizes[i] >= min_size:        
            img2[output == i + 1] = 255

    res = cv2.bitwise_not(img2)

    plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122),plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()
 

def noise_4():
    image = cv2.imread("1.png")
    # cv2.imshow("Image", image)
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # apply Otsu's automatic thresholding
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel)
    opening = cv2.GaussianBlur(opening, (5, 5), 0)
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 1 # threshhold value for small noisy components
    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        # print("2")
        if sizes[i] >= min_size:        
            img2[output == i + 1] = 255
    res = cv2.bitwise_not(img2)

    custom_config = r"-l tha -c tessedit_char_blacklist=~|!-<>@#%&_ --oem 3 --psm 6"
    text = pytesseract.image_to_string(res , config=custom_config) 
    text = re.sub(" ","", text)
    print(text)
    

    plt.subplot(121),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(122),plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()

# noise_4()
