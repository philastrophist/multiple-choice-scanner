from functools import partial
from math import floor

import cv2
import numpy as np
from matplotlib import pyplot as plt

from mark import k_means

# bgr
BLACK = (0, 0, 0)
GREEN = (50, 205, 50)
RED = (0, 0, 255)
BLUE = (255, 191, 0)
YELLOW = (0, 255, 255)


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


image = cv2.imread("images/corrections/Initial 2 - A-1 - marked.jpg")

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



def split_sections(grid, buffer=3):
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
                        sections.append(grid[(xclassify == kx) & (yclassify == ky)])
                if all(valid):
                    return sections
    raise ValueError(f"Cannot split into sections")


def regularise_grid(g):
    """
    sort into y-bins of equal size
    """
    points = g[:, :2]
    min_width = np.min(g[:, 2])

    topleft = np.min(points, axis=0) - (min_width // 2)
    bottomright = np.max(points, axis=0) + (min_width // 2)

    edges = None
    for nx in range(1, 100):
        ns, _edges = np.histogram(points[:, 0], nx, (topleft[0], bottomright[0]))
        if not all(ns):
            return edges
        edges = _edges
    raise ValueError(f"Cannot regularise grid")


def overlapped_1d(a, b):
    x1, w1 = a.T
    x2, w2 = b.T
    o1 = (x1 + w1 > x2) & (x1 < x2)
    o2 = (x2 + w2 > x1) & (x2 < x1)
    return (o1 | o2) | np.all(a == b, axis=0)

def contains(box, point):
    x, y, w, h = box
    xp, yp = point
    return x <= xp <= x+w and y <= yp <= y+h


def order_grid(g):
    """
    Takes a numpy array of shape (nboxes, (x, y, w, h)), orders them based on overlapping x,y coords
    and returns an array of shape (nrows, ncols, (x, y, w, h))
    """
    overlapped_y = np.zeros((g.shape[0], g.shape[0]), dtype=bool)
    overlapped_x = np.zeros((g.shape[0], g.shape[0]), dtype=bool)
    for i, box1 in enumerate(g):
        for j, box2 in enumerate(g):
            overlapped_y[i,  j] = overlapped_1d(box1[[1, 3]], box2[[1, 3]])
            overlapped_x[i,  j] = overlapped_1d(box1[[0, 2]], box2[[0, 2]])
    groups_y = sorted({tuple(np.where(i)[0]) for i in overlapped_y}, key=lambda d: g[d, 1].mean())  # [(a1, a2), (a1, a2), ...]
    groups_y = [[g[i] for i in sorted(group, key=lambda d: g[d, 0].mean())] for group in groups_y]
    means_x = sorted({np.mean(g[tuple(np.where(i)[0]), 0]) for i in overlapped_x})  # [(a1, a2), (a1, a2), ...]
    # now fill in missing boxes
    groups = []
    for group in groups_y:
        _group = []
        mean = np.mean(group, axis=0)
        for x in means_x:
            matched = [box for box in group if contains(box, (x, mean[1]))]
            if matched:
               _group.append(matched[0])
            else:
                _group.append((x, *mean[1:]))
        groups.append(_group)
    return np.asarray(groups, dtype=int)


def shrink_box(box, fraction=1.):
    """
    Takes an array of (x, y, w, h) and shrinks their size by %
    """
    x, y, w, h = box
    w1 = int(floor(w * fraction))
    h1 = int(floor(h * fraction))
    diffx = w - w1
    diffy = h - h1
    return x + (diffx // 2), y + (diffy // 2), w1, h1


def order_sections(sections):
    return sections


def cutout_box(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w]

def measure_box(image, box, shrink=0.5):
    box = shrink_box(box, shrink)
    return 1 - (cutout_box(image, box).mean() / 255)


def decide_row(binary_image, row, mistake_threshold=0.8, blank_threshold=0.1):
    """
    Mixtures of ⬛, ☐, ☒
        ⬛: mistake - ignore
        ☒: selected answer
        ☐: unselected answer

    Scenarios:
        ☒ ☐ ☐ ☐ one selected, rest blank -> valid
        ☒ ⬛ ⬛ ⬛ one selected, others mistake -> valid
        ⬛ ⬛ ⬛ ⬛ all mistake -> invalid
        ☒ ☒ ☒ ☒ all selected -> invalid

    Need to decide absolutely
        make binary image
        mistake = >80%
        blank = <10%
        select = in between
    """
    intensities = np.asarray(list(map(partial(measure_box, binary_image), row)))
    mistake = intensities > mistake_threshold
    blank = intensities < blank_threshold
    selected = ~mistake & ~blank
    if sum(mistake) == 1 and sum(selected) == 0:
        selected, mistake = mistake, selected
    valid = sum(selected) == 1
    if valid:
        i = np.where(selected)[0][0]
    elif all(blank):
        i = -2
    else:
        i = -1
    return i, valid

def binary_image(img):
    _, img_bin = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bin

def mark_section(binary_image, section, mistake_threshold=0.8, blank_threshold=0.1):
    answers, valids = zip(*[decide_row(binary_image, row, mistake_threshold, blank_threshold) for row in section])
    return np.asarray(answers), np.asarray(valids)


def present_section(image, section_boxes, student_answers, valid, correct_answers=None):
    if correct_answers is None:
        correct_answers = student_answers
    for r, (boxes, student, valid, correct) in enumerate(zip(section_boxes, student_answers, valid, correct_answers)):
        for c, box in enumerate(boxes):
            x, y, w, h = box
            if not valid and student != -2:
                colour = YELLOW
            elif student == correct == c:
                colour = GREEN
            elif student == c:
                colour = RED
            else:
                continue
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)




grid = np.asarray(grid)
print(grid.shape)
sections = split_sections(grid)
print(len(sections))

sections = list(map(order_grid, sections))
sections = order_sections(sections)

pass_perc = 0.8
score = 0
number = 0
invalids = 0
correct_answer_array = np.asarray([[0] * len(s) for s in sections])

for section, correct_answers in zip(sections, correct_answer_array):
    answers, valids = mark_section(binary_image(image), section)
    score += sum(answers == correct_answers)
    invalids += sum(~valids)
    number += len(answers)
    present_section(image, section, answers, valids, correct_answers)
passed = score / number >= pass_perc
str_passed = 'PASS' if passed else 'FAIL'
colour = GREEN if passed else RED

image = cv2.putText(image, f'{score:.0f} / {number:.0f} = {score / number:.1%} ({str_passed})', tuple(np.min(sections[0], axis=(0, 1))[:2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1., colour)
show_image('image', image)
cv2.waitKey(0)
