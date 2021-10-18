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


