# get digit predictions from my machine learning model

from tensorflow import keras
import cv2
import numpy as np

model = keras.models.load_model("digit_recognition_model.h5")


def get_numbers_array(sections_array):
    """
    Create an array of the digits in the puzzle entries from array of images
    :param sections_array: array of images of each cell of the puzzle
    :return: array of digits of each cell
    """
    digit_array = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            image = sections_array[i, j]/255
            image = cv2.resize(image, (50, 50))
            image = np.reshape(image, (1, 50, 50))
            if np.count_nonzero(image) > 30:
                digit_array[i, j] = int(np.argmax(model.predict(image, verbose=0))) + 1
            else:
                digit_array[i, j] = 0
    return digit_array.astype("int8")
