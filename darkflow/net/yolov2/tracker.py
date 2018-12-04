
# load the weights for the tracker
# this is for tracking method 1

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Flatten

class Tracker():

    def __init__(self):
        # weights_path = "C:\\Users\\grant\\Documents\\School\\Deep Learning\\Project_v2\\trunk\\tracking_weights\\tw1"
        weights_path = "C:\\fix\\this\\path\\Please!\\tw1"
        self.model = self.get_model()
        self.model.load_weights(weights_path)
        print("yolov2\\net\\tracker::__init__() -> tracking model made")


    def get_model(self):
        """ init and save as a variable a keras model object """
        num_of_classes = 2
        m = Sequential()
        # first hidden layer
        m.add(Conv2D(filters=40,
                    kernel_size=(10,10),
                    input_shape=(250,250,3), # check this
                    activation='relu'))
        # pooling layer
        m.add(
            MaxPooling2D(pool_size=(4,4))
        )
        # Flattening Layer
        m.add(
            Flatten()
        )
        #Fully connected layer
        m.add(
            Dense(
                units=200,
                activation='relu'
            )
        )
        #Output layer
        m.add(
            Dense(
                units=num_of_classes,
                activation='softmax'
            )
        )
        #compile model
        m.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        #return the model

        return m

    def get_prediction(self, img):
        """ processes the image from predict.py into a prediction for tracking. """
        # preproc image into format for the CNN
        # resize and scale pix values in all images
        # img = img.reshape(img.shape[0], 250 * 250 * 3).astype('float32')
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img.resize(250, 250, 3)

        img2 = np.expand_dims(img, axis=0)
        pred = self.model.predict_classes(img2)
        return pred
