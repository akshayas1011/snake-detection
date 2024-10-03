import numpy as np
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import cv2
import random
import sklearn.model_selection as model_selection
import datetime
from model import createModel
from contextlib import redirect_stdout
categories = ['Banded Racer','Checkered Keelback','Green Tree Vine','Common Rat Snake','Common Krait','King Cobra','Spectacled Cobra']
SIZE = 24
def getData():
    rawdata = []
    data = []
    dir = "./data/"
    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                rawdata = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_data = cv2.resize(rawdata, (SIZE, SIZE))

                data.append([new_data, class_num])
            except Exception as e:
                pass

    random.shuffle(data)

    img_data = []
    img_labels = []
    for features, label in data:
        img_data.append(features)
        img_labels.append(label)
    img_data = np.array(img_data).reshape(-1, SIZE, SIZE, 1)
    img_data = img_data / 255.0
    img_labels = np.array(img_labels)

    return img_data, img_labels



data, labels = getData()
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=0.20)

train_data, val_data, train_labels, val_labels = model_selection.train_test_split(train_data, train_labels,test_size=0.10)
print(len(train_data), " ", len(train_labels), len(test_data), " ", len(test_labels))

model = createModel(train_data)

checkpoint = keras.callbacks.ModelCheckpoint(filepath='./model/model1.h5', save_best_only=True, monitor='val_loss', mode='min')

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"], )


history = model.fit(train_data, train_labels, epochs=80
                    )

model.save('./model/model1.h5')
#test_loss, test_acc = model.evaluate(test_data, test_labels)
#print("Model Accuracy: ", test_acc, "Model Loss: ", test_loss)
