import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_data(classes, image_size=(30, 30)):
    data = []
    labels = []
    cur_path = os.getcwd()

    for i in range(classes):
        path = os.path.join('E:\\python\\BaiTapLon\\Train', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                image = Image.open(os.path.join(path, a))
                image = image.resize(image_size)
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except (IOError, OSError):
                print(f"Error loading image: {a}")

    data = np.array(data) / 255.0  
    labels = np.array(labels)
    return data, labels

def split_data(data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42, stratify=labels)
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    return X_train, X_test, y_train, y_test
