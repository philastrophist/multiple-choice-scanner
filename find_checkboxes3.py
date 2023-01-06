from collections import Counter
from functools import partial, reduce
from itertools import combinations
from math import floor

import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from feature_alignment import align_images
from mark import k_means

# bgr
BLACK = (0, 0, 0)
GREEN = (50, 205, 50)
RED = (0, 0, 255)
BLUE = (255, 191, 0)
YELLOW = (0, 255, 255)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
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


def show_image(name, img, width=800, colorspace=cv2.COLOR_GRAY2RGB, wait=True):
    resize = resize_with_aspect_ratio(img, width=width)
    cv2.imshow(name, resize)
    if wait:
        cv2.waitKey(0)


def coordinate_image():
    """
    Find features to identify whether this is a double sided image and what page.

    """


def warp_perspective(edged, *images):
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
    results = [four_point_transform(image, docCnt.reshape(4, 2)) for image in images]
    if len(results) == 1:
        return  results[0]
    return results

def get_grid(image, expected_ratio=None):  # expected_ratio=used to be 1.208
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)
    min_fill = np.percentile(canny, 1)

    # Find contours
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and draw rectangles around contours
      # area is 700 for a4
     # 5 for a4
    grid = []
    i = 0
    for c in cnts:
        r = cv2.boundingRect(c)
        x, y, w, h = r
        if expected_ratio is not None:
            buff = 1
            min_ratio = (h - buff) / (w + buff)
            max_ratio = (h + buff) / max([w - buff, 1])
            correct_size = (expected_ratio <= max_ratio) & (expected_ratio >= min_ratio)
        else:
            correct_size = True
        if correct_size:
            inner_half = shrink_box(r, 0.5)
            if np.prod(inner_half[-2:]) > 1:
                if (cutout_box(canny, inner_half) > min_fill).mean() < 0.01:
                    grid.append(r)
                    # cv2.rectangle(image, (x, y), (x + w, y + h), RED, 2)
                    # cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1., RED)
                    i += 1
    # show_image('im', image)
    return np.asarray(grid)


def split_sections(grid, buffer=3):
    points = grid[:, :-2]
    for nsections in [1] + list(range(2, 8, 1)):
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
    """
    Sort sections (nquestions, nanswers, (x, y, w, h)) by min((x,y)) so top to bottom, left to right
    """
    sections = [s for s in sections if len(s)]
    centroids = np.array([s.min(axis=(0, 1))[[0, 1]] for s in sections])
    sorti = np.lexsort((centroids[:, 0], centroids[:, 1]))
    return [sections[i] for i in sorti]


def cutout_box(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w]

def measure_box(image, box, shrink=1.):
    box = shrink_box(box, shrink)
    return 1 - (cutout_box(image, box).mean() / 255)


def decide_row(binary_image, row, mistake_threshold=0.8, blank_threshold=0.1, shrink=1.):
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
    intensities = np.asarray(list(map(partial(measure_box, binary_image, shrink=shrink), row)))
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
    if img.ndim == 3:
        _, img_bin = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return img_bin
    return img

def mark_section(binary_image, section, mistake_threshold=0.8, blank_threshold=0.1, shrink=1.):
    answers, valids = zip(*[decide_row(binary_image, row, mistake_threshold, blank_threshold, shrink) for row in section])
    return np.asarray(answers), np.asarray(valids)


def present_section(image, section_boxes, student_answers, valid, correct_answers=None, highlight_empty=False, shrink=1.):
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
            elif highlight_empty:
                colour = BLUE
            else:
                continue
            x, y, w, h = shrink_box(box, shrink)
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)


def cluster_boxes(grid, nclusters):
    """
    Gaussian mixture clustering in space of (nearest 3 neighbours, width, height)
    Scenarios:
        1. 1 spike for the grid, large scatter for others
        2. large scatter encompassing grid and others, 1 spike for others
    Winning scenario: the most populated is a spike containing only grid boxes
    Keep increasing
    """
    dist_matrix = cdist(grid[:, :2], grid[:, :2])
    nearest = np.sort(dist_matrix, axis=1)[:, 1:4]
    nearest /= nearest[:, :1]  # make it relative, first col will be 1.
    features = np.append(nearest[:, 1:], grid[:, 2:], axis=1)
    gm = GaussianMixture(n_components=nclusters, random_state=0).fit(features)
    predict = gm.predict(features)
    return predict, np.linalg.det(gm.covariances_), np.array([sum(predict == i) for i in range(nclusters)])


def test_separate(grid1, grid2):
    """
    find 3 shortest distances in each grid (should be horizontal, vertical, and diagonal)
    find shortest distance between grids
    if the maximum intra-grid distance is smaller than the minimum inter-grid distance, then they are separate
    """
    intra = np.max(np.median(np.sort(cdist(grid1[:, :2], grid1[:, :2]), axis=1)[:, 1:4], axis=0)) * 1.5
    inter = np.min(cdist(grid1[:, :2], grid2[:, :2]))
    return inter > intra


def test_consistent(grid):
    km = KMeans(n_clusters=2, random_state=0).fit(grid[:, :2])
    grid1, grid2 = grid[km.labels_ == 0], grid[km.labels_ == 1]
    plt.scatter(*grid1[:, :2].T)
    plt.scatter(*grid2[:, :2].T)
    plt.show()
    return not test_separate(grid1, grid2)


def break_into_sections(X):
    sections = [X]
    for nsections in range(2, 8):
        km = KMeans(n_clusters=nsections, random_state=0).fit(X[:, :2])
        _sections = [X[km.labels_ == i] for i in range(nsections)]
        is_valid = all(map(lambda x: test_separate(*x), combinations(_sections, 2))) and all(map(test_consistent, _sections))
        if is_valid:
            sections = _sections
        else:
            sections = list(map(order_grid, sections))
            sections = order_sections(sections)
            return sections
    raise ValueError(f"Cannot parse sections")


def extract_sections(image):
    """
    Cluster the boxes with similar boxes
    Try to make consistent grid sections out of them
    Choose the most populated cluster label if there are multiple valid ones
    Use the least clusters as possible
    """
    grid = get_grid(image)
    for nclusters in range(1, 5):
        labels, dets, ns = cluster_boxes(grid, nclusters)
        sections = []
        for cluster, occupancy in enumerate(ns):
            if occupancy < 4:
                continue
            try:
                _sections = break_into_sections(grid[labels == cluster])
                sections.append((occupancy, _sections))
            except ValueError:
                continue
        return max(sections, key=lambda x: x[0])[1]
    raise ValueError(f"Unable to parse grid")


def mark_image(sections, pass_perc, image, highlight_empty=False, shrink=0.5):
    score = 0
    number = 0
    invalids = 0
    correct_answer_array = [[0] * len(s) for s in sections]

    for section, correct_answers in zip(sections, correct_answer_array):
        answers, valids = mark_section(binary_image(image), section, shrink=shrink)
        score += sum(answers == correct_answers)
        invalids += sum(~valids)
        number += len(answers)
        present_section(image, section, answers, valids, correct_answers, highlight_empty, shrink=shrink)
    passed = score / number >= pass_perc
    str_passed = 'PASS' if passed else 'FAIL'
    colour = GREEN if passed else RED
    image = cv2.putText(image, f'{score:.0f} / {number:.0f} = {score / number:.1%} ({str_passed})', tuple(np.min(sections[0], axis=(0, 1))[:2]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1., colour)


def overlay_images(a, b):
    return cv2.addWeighted(a, 0.5, b, 0.5, 0)


if __name__ == '__main__':
    answer_image = cv2.imread("images/test-answer.jpeg")
    calibration_image = cv2.imread("images/corrections/Initial 2 - A-1.jpg")
    student_image = cv2.imread("images/corrections/Initial 2 - A-1 - marked.jpg")
    calibration_image = resize_with_aspect_ratio(calibration_image[:, calibration_image.shape[1]//2:], answer_image.shape[1])
    student_image = resize_with_aspect_ratio(student_image[:, student_image.shape[1]//2:], answer_image.shape[1])
    gray = cv2.cvtColor(answer_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)
    answer_image = warp_perspective(canny, answer_image)

    sections = extract_sections(calibration_image)

    mask = np.zeros_like(calibration_image)
    for section in sections:
        x0, y0 = section.min(axis=(0, 1))[:2]
        x1, y1 = section.max(axis=(0, 1))[:2] + section.max(axis=(0, 1))[2:]
        calibration_image[y0-1:y1+1, x0-1:x1+1] = 255
    h, (answer_image, calibration_image) = align_images(binary_image(answer_image), binary_image(calibration_image), answer_image, calibration_image, good_match_fraction=0.5)
    mark_image(sections, 0.8, answer_image, highlight_empty=False)
    show_image('answer', answer_image, wait=True)
    # overlay = overlay_images(answer_image, calibration_image)
    # show_image('overlay', overlay)
    # show_image('cal', calibration_image)



    # take calibration pdf
    # 4-point transform paper
    # attempt to align double-sided
    # attempt to align single-sided
    # read grid


    #
    # mark_image(sections, 0.8, student_image, 'marked')

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # canny = cv2.Canny(blurred, 120, 255, 1)
