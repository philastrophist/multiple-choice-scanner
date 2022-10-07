from __future__ import print_function
import cv2
import numpy as np
from mark import find_centres



def align_images(im1, im2, *args, good_match_fraction=0.15, max_features=500):
  """
  Align im1 to im2
  """

  # im1Gray = find_centres(im1)[-1]
  # im2Gray = find_centres(im2)[-1]

  # Convert images to grayscale
  # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(max_features)
  keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = list(matcher.match(descriptors1, descriptors2, None))

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * good_match_fraction)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width = im2.shape
  imRegs = [cv2.warpPerspective(i, h, (width, height)) for i in args]
  return h, imRegs

if __name__ == '__main__':

  # Read reference image
  refFilename = "images/test_01.png"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # Read image to be aligned
  imFilename = "images/test_02.png"
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  imReg, h, imRef = align_images(im, imReference)

  # Write aligned image to disk.
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename)
  # cv2.imwrite(outFilename, imReg)
  cv2.imshow("paper", imReg)
  cv2.imshow("reference", imRef)
  cv2.waitKey(0)

  # Print estimated homography
  print("Estimated homography : \n",  h)