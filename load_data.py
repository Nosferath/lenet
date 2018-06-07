import os
import cv2
import random
import numpy as np
import sklearn.model_selection as sk

def load_data(batch_size, image_dir):
    # Open images
    filenames_p = os.listdir(image_dir + 'p/')
    filenames_n = os.listdir(image_dir + 'n/')
    images = []
    labels = []
    for filename in filenames_p:
        images.append(cv2.imread(image_dir + 'p/' + filename))
        labels.append(1)
    for filename in filenames_n[0:len(filenames_p)]:
        images.append(cv2.imread(image_dir + 'n/' + filename))
        labels.append(0)
    # Shuffle images, as database.shuffle doesn't seem to work
    indexes = list(range(len(images)))
    random.shuffle(indexes)
    images = [images[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    print("%i imagenes cargadas."%len(images))
    # Converting list([98,98,3]) to array(n, 98, 98, 3)
    images = np.float32(np.stack(images))
    labels = np.int32(np.array(labels))
    images_train, images_eval, labels_train, labels_eval = sk.train_test_split(
        images, labels, test_size=0.3, random_state=42)
    return images_train, images_eval, labels_train, labels_eval