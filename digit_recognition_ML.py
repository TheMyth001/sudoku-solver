import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import cv2
import glob


# generating my own train/test images because
# machines that learn on MNIST database
# do not perform well outside the MNIST database
def generate_sample_image():
    """
    Generates a (50, 50) image of a random digit in a random font, font size, and position
    """
    digit = random.randint(0, 8)
    size = random.randint(180, 250)
    errX = random.randint(-10, 10)
    errY = random.randint(-10, 10)
    y = 128 - size/2.2 + errY
    x = 128 - size/3.5 + errX
    color = 255

    img = Image.new("L", (256, 256))
    draw = ImageDraw.Draw(img)
    font_face = random.choice(fonts)
    font = ImageFont.truetype(font_face, size)
    draw.text((x, y), str(digit+1), color, font=font)
    img = img.resize((50, 50), Image.BILINEAR)
    # noinspection PyTypeChecker
    img = cv2.GaussianBlur(src=np.asarray(img), ksize=(5, 5), sigmaX=0)
    return img, digit


# list of all fonts available on the computer
fonts = glob.glob("C:\\Windows\\Fonts" + "\\*.ttf")

# create training set
trainX, trainY = [], []
for i in range(20000):
    image, digit = generate_sample_image()
    image = image/255
    trainX.append(image)
    trainY.append(digit)

# create testing set
testX, testY = [], []
for i in range(1000):
    image, digit = generate_sample_image()
    # noinspection PyTypeChecker
    image = np.asarray(image)/255
    testX.append(image)
    testY.append(digit)

trainX = np.array(trainX)
testX = np.array(testX)

# converting train and test outputs to one-hot form to implement categorical model
trainY = tf.one_hot(
    indices=trainY,
    depth=9,
    on_value=1,
    off_value=0,
)
testY = tf.one_hot(
    indices=testY,
    depth=9,
    on_value=1,
    off_value=0,
)

# building the model
model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(50, 50, 1)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(32, (5, 5), activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(9, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["categorical_accuracy"],
)
model.fit(trainX, trainY, epochs=3)

# evaluating and saving as a file for future
print("\nTest set:")
model.evaluate(testX, testY)
model.save('digit_recognition_model_CNN.h5')
