import cv2

class DrawLineWidget(object):
    def __init__(self, img):
        self.original_image = img
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []
        self.lines = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates.append((x,y))

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[-2], self.image_coordinates[-1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[-2], self.image_coordinates[-1], (36,255,12), 2)
            cv2.imshow("image", self.clone)
            if self.image_coordinates[-2] != self.image_coordinates[-1]:
                self.lines.append(self.image_coordinates[-2:])

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone


def match_line_pairs(lines):
    from shapely.geometry import LineString

    for a in lines:
        for b in lines:
            if a != b:
                intersection = LineString(a).intersection(LineString(b))
                if not intersection.is_empty:
                    h, v = normalise_line_pairs(a, b)




def normalise_line_pairs(a, b):
    meana = np.mean(a, axis=0)
    meanb = np.mean(b, axis=0)
    if a[0][0] - a[1][0] > b[0][0] - b[1][0]:
        horizontal, vertical = [(a[0][0], meana[1]), (a[1][0], meana[1])], [(b[0][0], meanb[1]), (b[1][0], meanb[1])]
    else:
        horizontal, vertical = [(b[0][0], meanb[1]), (b[1][0], meanb[1])], [(a[0][0], meana[1]), (a[1][0], meana[1])]
    return horizontal, vertical



if __name__ == '__main__':
    import numpy as np
    img = np.zeros((512,512,3), np.uint8)
    draw_line_widget = DrawLineWidget(img)

    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)
        if key == 13:  # Close program with keyboard 'enter'
            cv2.destroyWindow('image')
            break


    clone = draw_line_widget.original_image.copy()
    cv2.imshow('image', clone)
    for line in draw_line_widget.lines:
        cv2.line(clone, *line, (255,20,147), 2)
    cv2.imshow('image', clone)
    while True:
        key = cv2.waitKey(1)
        if key == 13:  # Close program with keyboard 'enter'
            cv2.destroyWindow('image')
            break