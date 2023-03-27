import cv2
import imutils
from imutils import perspective
import numpy as np


def puzzle_extraction(file):
    # reading and preprocessing image
    original = cv2.imread(f"sudoku images/{file}")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)[1]

    # puzzle boundary recognition by finding the largest contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # approximating puzzle contour to a rectangle
    peri = cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, 0.01 * peri, True)

    # discarding the background using 4 point transform
    puzzle = perspective.four_point_transform(thresh, c.reshape(4, 2))

    # dividing the puzzle into 81 sections
    section_array = []
    puzzle = cv2.resize(puzzle, (900, 900))
    for i in range(9):
        for j in range(9):
            extract = puzzle[100*i+10:100*i+90, 100*j+10:100*j+90]
            thresh = cv2.threshold(extract, 150, 255, cv2.THRESH_BINARY)[1]
            resize = cv2.resize(thresh, (50, 50))
            section_array.append(cv2.GaussianBlur(src=resize, ksize=(5, 5), sigmaX=0))
    section_array = np.array(section_array).reshape((9, 9, 50, 50))

    return section_array
