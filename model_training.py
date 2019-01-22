import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle

import csv
import cv2
import numpy as np
from scipy import ndimage
from glob import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# Keras models layers
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from keras.layers import Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model

batch_size = 16
epochs = 1

CONTINUE = True
MULTI_GPU = False
GPU_MEMORY_SIZE_GigBYTES = 2
GPU_MEMORY_SIZE_BYTES = (GPU_MEMORY_SIZE_GigBYTES*(2**30))
#print(GPU_MEMORY_SIZE_BYTES)
Max_trainable_parameters = GPU_MEMORY_SIZE_BYTES/(4*batch_size)
print("Configured device internal memory %d Giga bytes" %GPU_MEMORY_SIZE_GigBYTES)
print("You can train up to %s parameters" %"{:,}".format(Max_trainable_parameters))
from common import *
import common
Model_function = "VN_model"
#Model_function = "pre_trained_model"
pre_trained_model = "VGG16"
#keras transfer learning
# VGG
if pre_trained_model == "VGG16":
    from keras.applications.vgg16 import VGG16 as Model_def
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions
elif pre_trained_model == "googlenet":
#Googlenet
    from keras.applications.inception_v3 import InceptionV3  as Model_def
    from keras.applications.inception_v3 import preprocess_input
    from keras.applications.inception_v3 import decode_predictions
elif pre_trained_model == "Resnet":
#Resnet Microsoft
    from keras.applications.resnet50 import ResNet50  as Model_def
    from keras.applications.resnet50 import preprocess_input
    from keras.applications.resnet50 import decode_predictions

if clip_image:
    image_hight -= removed_pixels

freeze_flag = False

#-------------
# data loading
#-------------




def seq_model():
    images, train_x_gray, train_x_normalized, train_x_cliped, train_y = load_images()

    train_x = images
    global model
    model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=(image_hight, image_width, image_depth), padding='SAME'),
    Dropout(0.25),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (5, 5), activation='relu', padding='SAME'),
    Dropout(0.25),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (5, 5), activation='relu', padding='SAME'),
    Dropout(0.25),
    MaxPooling2D(pool_size=(2, 2)),
    #Flatten(input_shape=(80, 160, 32)),
    Flatten(),
    Dense(128, activation = "relu"),
    Dropout(0.5),
    Dense(64, activation = "relu"),
    Dropout(0.5),
    Dense(1)]
    )
    print(model.summary())

    # Compile the model
    model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])

    model.fit(train_x, train_y, validation_split=.2, shuffle=True, batch_size=32, epochs=10)


def VN_model():
    global model
    #images, train_x_gray, train_x_normalized, train_x_cliped, train_y = load_images(image_depth=1, norm_image=True, clip_image=True, save=True)
    images, train_x_gray, train_x_normalized, train_x_cliped, train_y = load_images(image_depth=image_depth, norm_image=False, clip_image=True, save=True)

    print(np.max(images), np.min(images))
    print(np.max(train_x_gray), np.min(train_x_gray))
    print(np.max(train_x_normalized), np.min(train_x_normalized))
    print(np.max(train_x_cliped), np.min(train_x_cliped))
    print(np.max(train_y), np.min(train_y))

    train_x = train_x_cliped
    model = Sequential([
    Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(image_hight, image_width, image_depth)),
    #Conv2D(64, (5, 5), activation='relu', input_shape=(image_hight, image_width, image_depth), padding='SAME'),
    Conv2D(64, (5, 5), activation='relu', padding='SAME'),
    #Dropout(0.25),
    #MaxPooling2D(pool_size=(4, 4)),
    Conv2D(64, (5, 5), activation='relu', padding='SAME'),
    #Dropout(0.25),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(48, (5, 5), activation='relu', padding='SAME'),
    #Dropout(0.25),
    #MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='SAME'),
    #Dropout(0.25),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(24, (3, 3), activation='relu', padding='SAME'),
    #Dropout(0.25),
    #MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding='SAME'),
    Flatten(),
    #Dense(128, activation = "relu"),
    #Dropout(0.5),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1)]
    )
    model.summary()
    if CONTINUE:
        model.load_weights(filepath="./VN_model")

    if MULTI_GPU:
        parallel_model = multi_gpu_model(model, gpus=2)
    # Compile the model
    if MULTI_GPU:
        parallel_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    train_x, train_y = shuffle(train_x, train_y)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    if MULTI_GPU:
        parallel_model.fit(train_x, train_y, validation_split=.2, shuffle=True, batch_size=batch_size, epochs=epochs)
    else:
        model.fit(train_x, train_y, validation_split=.2, shuffle=True, batch_size=batch_size, epochs=epochs)

    #model.fit_generator(datagen.flow(train_x, train_y, batch_size=16, shuffle=True), epochs=5)


def pre_trained_model():
    global model

    images, train_x_gray, train_x_normalized, train_x_cliped, train_y = csv_load_images(image_depth=3, norm_image=False, clip_image=True, save=False)  # load_images()

    train_x = images

    #global model, train_x, train_y
    pretrained_model = Model_def(weights='imagenet', include_top=False, input_shape=train_x.shape[1:])
    ####  Freeze the loaded layer
    if freeze_flag == True:
        for layer in pretrained_model.layers:
            layer.trainable = False
    #out_1 = pretrained_model(Input(shape=(image_hight,image_width,3)))
    F = Flatten()(pretrained_model.layers[-1].output)
    x = Dense(512, name="new_dense1", activation = "relu")(F)
    d = Dropout(0.5)(x)

    #x = Dense(200, name="new_dense1", activation = "relu")(out_1)
    xx = Dense(256, name="new_dense2", activation = "relu")(d)
    dd = Dropout(0.5)(xx)

    xx = Dense(128, name="new_dense2", activation = "relu")(d)
    dd = Dropout(0.5)(xx)

    xxx = Dense(1, name="new_dense3")(dd)
    model = Model(input=pretrained_model.input, output=xxx)
    model.summary()

    # Compile the model
    model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    #vgg16_predictions = pretrained_model.predict(train_x)
    train_x = preprocess_input(train_x)
    train_x, train_y = shuffle(train_x, train_y)

    # Train the model
    # Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time
    '''model.fit_generator(datagen.flow(X_train, y_one_hot_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs, verbose=1,
                        validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size),
                        validation_steps=len(X_val) / batch_size)'''

    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_split=.2, shuffle=True)

def main():
    #VN_model()
    exec(Model_function+"()")
    #pre_trained_model()
    if Model_function == "pre_trained_model":
        Model_name_save = pre_trained_model
    else:
        Model_name_save = Model_function
    print("Saving Model: " + Model_name_save)
    model.save(Model_name_save)
    #model.loa
    # Open the file
    with open(Model_name_save + '_report.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    plot_model(model, to_file=Model_name_save+'.png', show_shapes=True, show_layer_names=True)

# -------------------------------------
# Entry point for the script
# -------------------------------------
if __name__ == '__main__':
    main()