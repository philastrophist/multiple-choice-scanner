import matplotlib.pyplot as plt
import numpy as np

from find_checkboxes3 import *
import networkx as nx


def find_consistent_boxes(X):
    dists = np.zeros((len(X), len(X)), dtype=bool)
    for i, box1 in enumerate(X):
        for j, box2 in enumerate(X):
            if i == j:
                dists[i, j] = True
            spec_dist = (box1[2:] - box2[2:])
            dists[i, j] = np.sum(spec_dist**2.) <= 2
    count_similar = np.sum(dists, axis=0)
    return X[np.where(count_similar == count_similar.max())[0]]

def find_consistent_separations(X):
    xdists = cdist(X[:, :1], X[:, :1])
    ydists = cdist(X[:, 1:2], X[:, 1:2])
    xdists[xdists <= 1] = np.nan
    ydists[ydists <= 1] = np.nan
    x = Counter(np.nanmin(xdists, axis=0)).most_common(1)[0][0]
    y = Counter(np.nanmin(ydists, axis=0)).most_common(1)[0][0]
    return x, y


def factors(n, start=1):
    return set(reduce(list.__add__, ([i, n//i] for i in range(start, int(n**0.5) + 1) if n % i == 0)))

def factor_pairs(n, start=1):
    fs = factors(n, start)
    return {(f, n//f) for f in fs}

# def optimise_grid(X, xlims, ylims, nlimit=10_000):
    # """Given a set bounding rect, find the lowest number of bins such that each bins is either empty
    # or has 1 occupant"""
    # for n in range(X.size, nlimit):
    #     for nx, ny in factor_pairs(n, 2):
    #         hist, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], (nx, ny), [xlims, ylims])
    #         if np.all(hist <= 1):
    #             xbins = np.digitize(X[:, 0], xedges)
    #             ybins = np.digitize(X[:, 1], yedges)
    #             return xbins, ybins, (nx, ny)
    # raise ValueError(f"Cannot find a grid < {nlimit} cells which encompasses the data")

def optimise_bins(X, lims, start, nlimit=1000):
    prev_edges = lims
    for n in range(start, nlimit):
        hist, edges = np.histogram(X, n, lims)  # last bin is inclusive interval, all others are right-exclusive
        if np.any(hist == 0):  # return previous bins when there is an empty bin in the current ones
            return prev_edges
        prev_hist, prev_edges = hist, edges
    raise ValueError(f"Cannot find a grid < {nlimit} cells which encompasses the data")

def optimise_grid(X, xlims, ylims, nlimit=1000):
    """Given a set bounding rect, find the lowest number of bins such that each bins is either empty
       or has 1 occupant"""
    xbins = optimise_bins(X[:, 0], xlims, 2, nlimit)  # there must be at least 2 questions
    ybins = optimise_bins(X[:, 1], ylims, 1, nlimit)
    return xbins, ybins




def infer_grids(X):
    x, y = find_consistent_separations(X)
    dists = cdist(X[:, :2], X[:, :2])
    maxdiag = np.sqrt((x+1)**2 + (y+1)**2)
    mindiag = np.sqrt((x-1)**2 + (y-1)**2)
    diagmask = (dists < maxdiag) & (mindiag < dists)
    xmask = (x+1 > dists) & (dists > x-1)
    ymask = (y+1 > dists) & (dists > y-1)
    legal = np.where(diagmask | xmask | ymask)
    G = nx.Graph()
    G.add_edges_from(zip(*legal))
    islands = map(list, nx.connected_components(G))
    ordered_islands = []
    for island in islands:
        points = X[island]
        w, h = X[island][:, 2:].mean(axis=0).astype(int)
        xlims = points[:, 0].min() - w/2, points[:, 0].max() + w/2
        ylims = points[:, 1].min() - h/2, points[:, 1].max() + h/2
        xedges, yedges = optimise_grid(points, xlims, ylims)
        xbinned = np.digitize(points[:, 0], xedges)
        ybinned = np.digitize(points[:, 1], yedges)
        xbins = [points[xbinned == b, 0].mean() for b in range(0, len(xbinned)+1) if np.any(xbinned == b)]
        ybins = [points[ybinned == b, 1].mean() for b in range(0, len(ybinned)+1) if np.any(ybinned == b)]
        grid = np.asarray([[[x, y, w, h] for x in xbins] for y in ybins])
        ordered_islands.append(grid)
    ordered_islands.sort(key=lambda x: tuple(x[0, 0, :2]))
    return ordered_islands


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

    grid = find_consistent_boxes(get_grid(calibration_image))
    print(grid.shape)

    grid = grid.tolist()
    del grid[0]
    del grid[5]
    del grid[6]
    del grid[7]
    del grid[8]
    del grid[20]
    del grid[21]
    del grid[22]
    grid = np.asarray(grid)
    islands = infer_grids(grid)
    print(islands)
    plt.scatter(*grid[:, :2].T, s=1, c='k')
    plt.scatter(*islands[0][0, :, :2].T)
    plt.scatter(*islands[1][1, :, :2].T)
    # for island in islands:
    #     for i, point in enumerate(island):
    #         print(point[0], point[1], str(i))
    #         plt.text(point[0], point[1], str(i), )
    plt.show()