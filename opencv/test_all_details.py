"""
    Some codes are from somewhere, I couldn't remember but finding the original version that i got. 
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


