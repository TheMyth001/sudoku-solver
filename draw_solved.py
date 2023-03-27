# for drawing the solved digits on the original image

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def draw(file, changes_array):
    """
    Drawing the solutions on the puzzle
    :param file: image file of the sudoku puzzle
    :param changes_array: array of changes to be added to the puzzle
    :return: display the solved puzzle
    """
    image = cv2.imread(f"sudoku images/{file}")
    image = cv2.resize(image, (450, 450))
    image = Image.fromarray(np.uint8(image))
    drawing = ImageDraw.Draw(image)
    size = 25
    for i in range(9):
        for j in range(9):
            if changes_array[i, j] != 0:
                y = 50*i + 25 - size/2.2
                x = 50*j + 25 - size/3.5
                font_face = "C:\\Windows\\Fonts\\Arial.ttf"
                font = ImageFont.truetype(font_face, size)
                drawing.text((x, y), str(changes_array[i, j]), (100, 100, 100), font=font)

    # noinspection PyTypeChecker
    cv2.imshow("Solved", np.asarray(image))
    cv2.waitKey()
