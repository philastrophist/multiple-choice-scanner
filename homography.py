import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from find_checkboxes3 import resize_with_aspect_ratio, binary_image, extract_sections, show_image, mark_image

MIN_MATCH_COUNT = 10
answer_image = cv.imread("images/test-answer.jpeg")
calibration_image = cv.imread("images/corrections/Initial 2 - A-1.jpg")
student_image = cv.imread("images/corrections/Initial 2 - A-1 - marked.jpg")
calibration_image = resize_with_aspect_ratio(calibration_image[:, calibration_image.shape[1] // 2:], answer_image.shape[1])
student_image = resize_with_aspect_ratio(student_image[:, student_image.shape[1] // 2:], answer_image.shape[1])
img1 = binary_image(calibration_image)
img2 = binary_image(answer_image)
sections = extract_sections(calibration_image)
mask = np.zeros_like(calibration_image)
for section in sections:
    x0, y0 = section.min(axis=(0, 1))[:2]
    x1, y1 = section.max(axis=(0, 1))[:2] + section.max(axis=(0, 1))[2:]
    img1[y0 - 1:y1 + 1, x0 - 1:x1 + 1] = 255

import time
start = time.perf_counter()

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
height, width = img1.shape
warped = cv.warpPerspective(answer_image, np.linalg.inv(M), (width, height))

mark_image(sections, 0.8, warped, highlight_empty=True)

end = time.perf_counter()
print(f"Finished in {end-start:.2f} seconds")
show_image('img', img3, wait=False)
show_image('warped', warped, wait=True)
