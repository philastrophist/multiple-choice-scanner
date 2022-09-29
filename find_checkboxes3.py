import cv2
import numpy as np
from matplotlib import pyplot as plt

from mark import k_means


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


def show_image(name, img, colorspace=cv2.COLOR_GRAY2RGB):
    resize = ResizeWithAspectRatio(img, width=1280)
    cv2.imshow(name, resize)
    cv2.waitKey(0)


image = cv2.imread("images/corrections/Initial 2 - A-1.jpg")

# image += np.random.normal(0, 0.2, size=image.shape).astype(image.dtype)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred, 120, 255, 1)

# Find contours
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Iterate thorugh contours and draw rectangles around contours
expected_area_fraction = 0.00018  # area is 700 for a4
threshold_area_fraction = 1.3e-06 # 5 for a4

size = np.prod(image.shape[:-1])
expected_area = expected_area_fraction * size
threshold = threshold_area_fraction * size

grid = []
for c in cnts:
    r = cv2.boundingRect(c)
    x, y, w, h = r
    area = w * h
    if (area > expected_area - threshold) & (area < expected_area + threshold):
        grid.append(r)


# fill missing grid elements
    # make grid with extremes of (x,y) detections
    # transform grid with translation and rotations to get nearest to all points



def split_sections(grid, imshape, buffer=3):
    points = grid[:, :-2]
    for nsections in [1] + list(range(2, 8, 2)):
        for nx in range(1, nsections+1):
            for ny in range(1, nsections+1):
                if nx * ny != nsections:
                    continue
                xclassify, xcentre = k_means(points[:, :1], nx)
                xsort = np.argsort(xcentre[:, 0])
                xclassify, xcentre = xsort[xclassify], xcentre[xsort]

                yclassify, ycentre = k_means(points[:, 1:], ny)
                ysort = np.argsort(ycentre[:, 0])
                yclassify, ycentre = ysort[yclassify], ycentre[ysort]

                valid = []
                sections = []
                for kx in range(0, nx):
                    for ky in range(0, ny):
                        ps = points[(xclassify == kx) & (yclassify == ky)]
                        xs = np.sort(ps[:, 0])
                        deltaxs = xs[1:] - xs[:-1]
                        ys = np.sort(ps[:, 1])
                        deltays = ys[1:] - ys[:-1]
                        validx = sorted(set(deltaxs))  # close together
                        validx = len(validx) - sum([abs(a - b) < buffer for a, b in zip(validx[:-1], validx[1:])]) <= 2
                        validy = sorted(set(deltays))  # close together
                        validy = len(validy) - sum([abs(a - b) < buffer for a, b in zip(validy[:-1], validy[1:])]) <= 2
                        valid.append(validx and validy)
                        sections.append(ps)
                if all(valid):
                    return sections
    raise ValueError(f"Cannot split into sections")



def discretise(points, min_width):
    topleft = np.min(points, axis=0) - (min_width // 2)
    bottomright = np.max(points, axis=0) + (min_width // 2)
    edges = None
    for nx in range(1, 100):
        ns, _edges = np.histogram(points[:, 0], nx, (topleft[0], bottomright[0]))
        if not all(ns):
            return edges
        edges = _edges
    raise ValueError(f"Cannot discretise grid")

grid = np.asarray(grid)
print(grid.shape)
sections = split_sections(grid, image.shape)
print(len(sections))
# xs = np.sort(grid[:, 0])
# deltaxs = xs[1:] - xs[:-1]
# ns, edges = np.histogram(deltaxs, np.arange(0, image.shape[0]+10, 1))
# print(edges[:-1][ns != 0])

cv2.line(image, (1892, 0), (1892, 2000), (36, 255, 12))
for i, r in enumerate(sections[1]):
    x, y = r
    w, h = 10, 10
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

show_image('image', image)
cv2.waitKey(0)
