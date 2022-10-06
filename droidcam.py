import imutils
from imutils.perspective import four_point_transform
import numpy as np
import cv2

cap = cv2.VideoCapture('http://192.168.0.13:4747/video?1280x720')

while(True):
    ret, image = cap.read()
    cv2.imshow('frame', image)
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
                # apply a four point perspective transform to both the
                # original image and grayscale image to obtain a top-down
                # birds eye view of the paper
                paper = four_point_transform(image, docCnt.reshape(4, 2))
                warped = four_point_transform(gray, docCnt.reshape(4, 2))
                cv2.imshow('detect', warped)
                break
    if docCnt is None:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()