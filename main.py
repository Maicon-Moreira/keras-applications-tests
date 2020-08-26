import tensorflow.keras.applications as apps
import numpy as np
from labels import labels
from matplotlib import image
from PIL import Image

image = Image.open('1.jpg')

model = apps.EfficientNetB7()


x = np.array(image)

y = model.predict(np.array([x]))[0]


def print_result(y):
    already_printed = []

    for _ in range(100):
        max_value = -100
        max_name = ''
        max_i = -100

        for i in range(1000):
            if i not in already_printed:
                if y[i] > max_value:
                    max_value = y[i]
                    max_name = labels[i]
                    max_i = i

        already_printed.append(max_i)

        print(str(max_value).ljust(15), str(max_name))


print_result(y)

model.save('model.h5')