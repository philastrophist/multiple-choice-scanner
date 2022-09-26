from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def find_centres(image):
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
    blur = cv2.GaussianBlur(warped, (3, 3), 0)
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
    centres.sort(key=lambda x: x[1])
    return np.array(centres), paper


def k_means(X, k, centers=None, num_iter=100):
    if centers is None:
        rnd_centers_idx = np.random.choice(np.arange(X.shape[0]), k, replace=False)
        centers = X[rnd_centers_idx]
    for _ in range(num_iter):
        distances = np.sum(np.sqrt((X - centers[:, np.newaxis]) ** 2), axis=-1)
        cluster_assignments = np.argmin(distances, axis=0)
        for i in range(k):
            msk = (cluster_assignments == i)
            centers[i] = np.mean(X[msk], axis=0) if np.any(msk) else centers[i]

    return cluster_assignments, centers


def build_detection_grid(centres, ncolumns, nsections):
    """
    divide into groups with ncolumns x nsections
    build grid in each
    returns a list of grids for each section
    """
    assignments, _ = k_means(centres, ncolumns * nsections)
    d = []
    for part in range(ncolumns * nsections):
        cs = centres[assignments == part]
        mean = np.mean(cs, axis=0)
        delta = cs[1:, :] - cs[:-1, :]
        n = np.round((cs[-1, :] - cs[0, :]) / delta)

        deltax = np.min(delta[:, 0])
        buffx = deltax / 2.
        deltay = np.min(delta[:, 1])
        buffy = deltay / 2.
        xedges = np.linspace(cs[0, 0]-buffx, cs[-1, 0]+buffx, n[0])
        yedges = np.linspace(cs[0, 1]-buffy, cs[-1, 1]+buffy, n[1])
        d.append(tuple(mean), xedges, yedges)
    d.sort(key=lambda x: x[0])
    return [i[1:] for i in d]


def detect(image):
    centres, masked = find_centres(image)
    # cutout and sum


if __name__ == '__main__':
    # 1. calibrate - get positions/bounding boxes of possible answers
    # 2. break into sections, make calibrated grid per section
    # 3. read in student
    # 4. align grid
    # 3. find max bit per row/section, this is the answer