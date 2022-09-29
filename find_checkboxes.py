import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import gcsfs



def count_edges_of_contour(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    return len(approx)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def show_image(img, colorspace=cv2.COLOR_GRAY2RGB):
    resize = ResizeWithAspectRatio(img, width=1280)
    cv2.imshow('image', resize)
    cv2.waitKey(0)


def imread_wrapper(uri):
    if uri.startswith("gs://"):
        with gcsfs.GCSFileSystem().open(uri, "rb") as f:
            arr = np.asarray(bytearray(f.read()), dtype="uint8")
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    else:  # Assume local file
        return cv2.imread(uri, cv2.IMREAD_COLOR)


img = imread_wrapper("images/corrections/Initial 2 - A-1.jpg")#"gs://fuzzylabs-jupyter-delicacies/form_segment.png")
show_image(img, colorspace=cv2.COLOR_BGR2RGB)


# Image preparation
#
# Three methods of preprocessing are investigated:
#
#     Thresholding (with Otsu's method to determine an optimal threshold)
#     Opening on the thresholded image. Modifiable parameter: KERNEL_LENGTH
#     Canny edge detector. Modifiable parameter: CANNY_KERNEL_SIZE

# Image thresholding
_, img_bin = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin = 255 - img_bin
print("Thresholded image")
show_image(img_bin)

# Opening with vertical kernel
KERNEL_LENGTH = 15
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,KERNEL_LENGTH))
vertical = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
print("Vertical opened image")
show_image(vertical)

# Opening with horizontal kernel
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_LENGTH,1))
horizontal = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
print("Horizontal opened image")
show_image(horizontal)

# Combining openings
img_opened = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0.0)
_, img_opened = cv2.threshold(img_opened, 0, 255, cv2.THRESH_BINARY)
print("Combined opened image")
show_image(img_opened)

#
CANNY_KERNEL_SIZE = 100
print("Canny edge detection")
img_canny = cv2.Canny(img, CANNY_KERNEL_SIZE, CANNY_KERNEL_SIZE)
show_image(img_canny, cv2.COLOR_BGR2RGB)

# Contours

def draw_contours(contours):
    show_image(cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 1), cv2.COLOR_BGR2RGB)

def get_contours(img_bin):
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_contours(contours)
    return contours

print("Thresholded image")
thresholded_contours = get_contours(img_bin)
print("Opened image")
opened_contours = get_contours(img_opened)
print("Canny edge detection")
edged_contours = get_contours(img_canny)

# Contour Filtering
#
# Multiple methods to filter the resulting contours

def are_bounding_dimensions_correct(contour, expected_area=625, tolerance=200, squareness_tolerance=5):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    return abs(area - expected_area) <= tolerance and abs(w - h) <= squareness_tolerance

def is_contour_square(contour, contour_tolerance=0.0015, square_side=25, area_tolerance=200):
    expected_area = square_side * square_side
    area = cv2.contourArea(contour)
    template = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]], dtype=np.int32)
    return cv2.matchShapes(template, contour, 3, 0.0) <= contour_tolerance and abs(area - expected_area) <= area_tolerance

def is_contour_rectangular(contour, contour_tolerance=1, expected_area=None, area_tolerance=None):
    rect = cv2.boundingRect(contour)
    area = rect[-2] * rect[-1]
    r = cv2.matchShapes(rect, contour, 3, 0.) <= contour_tolerance
    if expected_area is not None:
        return r and abs(area - expected_area) <= area_tolerance
    return r

def filter_contours(contours, func):
    _contours = [x for x in contours if func(x)]
    draw_contours(_contours)
    return _contours

cs = filter_contours(edged_contours, is_contour_rectangular)

# Filter by contour area, parameters: expected_area and tolerance

def is_correct_area(contour, expected_area=625, tolerance=300):
    area = cv2.contourArea(contour)
    return abs(area - expected_area) <= tolerance






# print("Thresholded image")
# filter_contours(thresholded_contours, is_correct_area)
# print("Opened image")
# filter_contours(opened_contours, is_correct_area)
# print("Canny edge detection")
# filter_contours(edged_contours, is_correct_area)

# Evidently, this filtering is not robust enough. We also want only the square boxes to be detected
#



# print("Thresholded image")
# filter_contours(thresholded_contours, are_bounding_dimensions_correct)
# print("Opened image")
# filter_contours(opened_contours, are_bounding_dimensions_correct)
# print("Canny edge detection")

# The previous method still picks up the circle as a valid check box
#
# Filtering using shape matching, parameter -- tolerance
#
# The template can be modified to catch shapes different from a 25x25 square.
#
# Filtering by assessing bounding boxes for squareness. Parameters: expected_area, tolerance and squareness_tolerance

# print("Thresholded image")
# filter_contours(thresholded_contours, is_contour_square)
# print("Opened image")
# filter_contours(opened_contours, is_contour_square)
# print("Canny edge detection")
# filter_contours(edged_contours, is_contour_square)
#
