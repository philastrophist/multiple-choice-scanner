from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


image = cv2.imread('images/debug.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
blur = cv2.GaussianBlur(warped,(3,3),0)
thresh = cv2.threshold(blur, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 10 and h >= 10 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

centres = []
for c in questionCnts:
    cv2.drawContours(paper, [c], -1, 255, -1)
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centres.append((cX, cY))
centres.sort(key=lambda x:x[1])
centres = np.array(centres)

thickness = 5
bound_x1 = [70, 170]
bound_x2 = [245, 350]
bound_y1 = [84, 330]
bound_y2 = [385, 465]
n1 = 30
n2 = 10
nc = 4

for l in [bound_x1, bound_x2, bound_y1, bound_y2]:
    l[0] -= thickness
    l[1] += thickness

_n1 = n1+1 if n1 % 2 else n1
_n2 = n2+1 if n2 % 2 else n2

cv2.line(paper, (bound_x1[0], bound_y1[0]), (bound_x1[1], bound_y1[0]), (0, 255, 0), thickness=5)
cv2.line(paper, (bound_x2[0], bound_y2[0]), (bound_x2[1], bound_y2[0]), (0, 255, 0), thickness=5)

cv2.line(paper, (bound_x1[0], bound_y1[1]), (bound_x1[1], bound_y1[1]), (0, 255, 0), thickness=5)
cv2.line(paper, (bound_x2[0], bound_y2[1]), (bound_x2[1], bound_y2[1]), (0, 255, 0), thickness=5)


import pandas as pd

i = slice(None)#0, 1)

partx1 = pd.cut(centres[i, 0], np.linspace(*bound_x1, num=nc+1), labels=['A', 'B', 'C', 'D'])
partx2 = pd.cut(centres[i, 0], np.linspace(*bound_x2, num=nc+1), labels=['A', 'B', 'C', 'D'])
party1 = pd.cut(centres[i, 1], np.linspace(*bound_y1, num=_n1//2+1), labels=range(1, _n1//2+1))
party2 = pd.cut(centres[i, 1], np.linspace(*bound_y2, num=_n2//2+1), labels=range(1, _n2//2+1))

sec1l = pd.DataFrame({'q': np.array(party1), 'a': partx1}).dropna(how='any').set_index('q')
sec1r = pd.DataFrame({'q': np.array(party1) + 15, 'a': partx2}).dropna(how='any').set_index('q')
sec1 = pd.concat([sec1l, sec1r])

sec2l = pd.DataFrame({'q': np.array(party2), 'a': partx1}).dropna(how='any').set_index('q')
sec2r = pd.DataFrame({'q': np.array(party2) + 15, 'a': partx2}).dropna(how='any').set_index('q')
sec2 = pd.concat([sec2l, sec2r])

print('Section 1\n=================')
print(sec1)
print('Section 2\n=================')
print(sec2)

cv2.imshow("paper", paper)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
