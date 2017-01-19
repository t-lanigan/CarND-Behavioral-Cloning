import numpy as np
import pandas as pd
import json
import time

import cv2

from keras.layers.core import Dense, Flatten, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.objectives import mse

from sklearn.model_selection import train_test_split
from scipy.misc import imread

TARGET_SIZE = (64, 64)

def read_imgs(img_paths):
    """
    read images from file path.
    """
    imgs = np.empty([len(img_paths), 64, 64, 3])

    for i, path in enumerate(img_paths):
        imgs[i] = preprocess_image(imread(path))

    return imgs


def resize_to_target_size(image):

    return cv2.resize(image, TARGET_SIZE)


def crop_and_resize(image):
    '''
    :param image: The input image of dimensions 160x320x3
    :return: Output image of size 64x64x3
    '''
    # Crop image
    cropped_image = image[55:135, :, :]
    processed_image = resize_to_target_size(cropped_image)
    return processed_image


def preprocess_image(image):
    image = crop_and_resize(image)
    image = image.astype(np.float32)

    # Normalize image
    image = image / 255.0 - 0.5
    return image


def get_model():

    model = Sequential([

        # Conv 5x5
        Convolution2D(24, 5, 5, border_mode='same', activation='elu',
                      input_shape=(64, 64, 3)),
        MaxPooling2D(border_mode='same'),
        # Conv 5x5
        Convolution2D(36, 5, 5, border_mode='same', activation='elu'),
        MaxPooling2D(border_mode='same'),
        # Conv 3x3
        Convolution2D(48, 5, 5, border_mode='same', activation='elu'),
        MaxPooling2D(border_mode='same'),
        # Conv 3x3
        Convolution2D(64, 3, 3, border_mode='same', activation='elu'),
        MaxPooling2D(border_mode='same'),
        SpatialDropout2D(0.2),
        # Conv 3x3
        Convolution2D(64, 3, 3, border_mode='same', activation='elu'),
        MaxPooling2D(border_mode='same'),
        SpatialDropout2D(0.2),
        # Flatten
        Flatten(),
        # Fully Connected
        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1, activation='linear')
    ])
    return model


def get_generator(imgs, angles, batch_size, flip_prob=0.6):
    """
    Generates random batches of the input data.
    :imgs: The input image file paths
    :angles: The steering angles associated with each image.
    :batch_size: The size of each minibatch.
    :yields A tuple (images, angles), where both images and
     angles have batch_size elements.
    """
    num_imgs = len(imgs)

    while True:

        indices = np.random.choice(num_imgs, batch_size)
        batch_imgs, angles_raw = read_imgs(imgs[indices]), angles[indices].astype(float)
        # Add random flipping for images
        for i in range(batch_size):
            prob = np.random.random()
            if prob < flip_prob:
                batch_imgs[i] = cv2.flip(batch_imgs[i], 1)
                angles_raw[i] = -angles_raw[i]

        yield batch_imgs, angles_raw


def save_model(model):
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    model.save_weights('model.h5')

    print('Model saved!')

    return True


def main():

    # 1. Read Data from Driving Log
    driving_log = pd.read_csv('driving_log.csv', index_col=False)
    driving_log.columns = ['center_imgpath', 'left_imgpath',
                           'right_imgpath', 'angle', 'throttle',
                           'break', 'speed']

    # 2. Prepare Data for Generator
    X_train_path, y_tr = [], []
    for index, row in driving_log.iterrows():
        # Include center image and steering angle.

        flip_a_coin = np.random.randint(2)
        if flip_a_coin != 2:
            X_train_path.append(row['center_imgpath'])
            y_tr.append(row['angle'])

        C = row['angle']
        L = C + 0.17
        R = C - 0.17

        X_train_path.append(row['left_imgpath'].strip())
        y_tr.append(L)

        X_train_path.append(row['right_imgpath'].strip())
        y_tr.append(R)

    X_train_path, y_tr = np.array(X_train_path), np.array(y_tr)

    start = time.clock()

    model = get_model()
    # plot(model, to_file='model.png', show_shapes=True)
    model.compile(optimizer='adam', loss='mse')

    X_train, X_val, y_train, y_val = train_test_split(X_train_path, y_tr,
                                                      test_size=0.10,
                                                      random_state=0)

    # Define Optimizer & Loss Function
    def custom_loss(y_true, y_pred):
        # loss is proportional to turning angle to reduce bias of straight
        # paths
        return mse(y_true, y_pred) * np.absolute(y_true)

    model.compile(loss=custom_loss, optimizer='adam')

    batch_size = 128

    model.fit_generator(get_generator(X_train, y_train, batch_size),
                        samples_per_epoch=12800,
                        validation_data=get_generator(X_val, y_val, batch_size),
                        nb_val_samples=len(y_val),
                        nb_epoch=5
                        )

    # Save Model/Weights
    save_model(model)

    end = time.clock()
    print("Training completed in {:2f}s".format(end - start))


if __name__ == '__main__':
    main()
