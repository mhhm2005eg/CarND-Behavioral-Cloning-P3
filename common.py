import pickle
import os
import numpy as np
from scipy import ndimage
import csv
import gc

data_path = "../CarND-Behavioral-Cloning-P3_data/data/"
drive_log_file = data_path + "driving_log.csv"
images_dir = data_path + "IMG/"
image_depth = 3
clip_image = True
norm_image = False
removed_pixels = 50
image_width = 320
image_hight = 160


def simulation_preprocesss(image_sample, image_depth=image_depth, norm_image=norm_image, clip_image=clip_image):
    # Gray scale
    if image_depth == 1:
        print("Convert to gray scale ...")
        train_x_gray = np.sum(image_sample / 3, axis=2, keepdims=True)

        # print(train_x.shape)
        # print(len(images))
        # exit(0)
        # Normalize
        if norm_image:
            print("Normalization ...")
            train_x_normalized = (train_x_gray/255) -0.5
        else:
            train_x_normalized = train_x_gray
    else:
        train_x_gray = None
        if norm_image:
            print("Normalization ...")
            train_x_normalized = (image_sample/255) - 0.5
        else:
            train_x_normalized = image_sample


    # Clip images
    if clip_image:
        print("Data Clipping ...")
        train_x_cliped = train_x_normalized[removed_pixels:, :, :]
        #image_hight -= removed_pixels
    else:
        train_x_cliped = train_x_normalized
        # print(train_x.shape)

    return  train_x_cliped

def save_nn(dist_pickle, samples_folder=data_path):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    data_file = "./"+samples_folder+"data.p"
    print("writing file : ", data_file)
    gc.collect()
    pickle.dump(dist_pickle, open(data_file, "wb"))


def loc_load_nn_file(file_path):
    # Read in the saved objpoints and imgpoints
    print("loading file: " + file_path)
    dist_pickle = pickle.load(open(file_path, "rb"))
    images = dist_pickle.get("images")
    train_x_gray = dist_pickle.get("train_x_gray")
    train_x_normalized = dist_pickle.get("train_x_normalized")
    train_x_cliped = dist_pickle.get("train_x_cliped")
    train_y = dist_pickle.get("train_y")
    return images, train_x_gray, train_x_normalized, train_x_cliped, train_y


def load_nn_file(folder_name):
    # Read in the saved objpoints and imgpoints
    file_path = folder_name+"/data.p"
    if os.path.isfile(file_path):
         return loc_load_nn_file(file_path)
    return np.array([None]), np.array([None])


def csv_load_images(image_depth=image_depth, norm_image=norm_image, clip_image=clip_image,save=True):
    global image_hight
    lines = []
    with open(drive_log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    steering_measurements = []
    index = 0
    for line in lines:
        if not index:
            index += 1
            continue
        source_path = line[0]
        #print(source_path)
        source_path = source_path.replace("\\", "/")
        #print(source_path)
        file_name = source_path.split('/')[-1]
        #print(file_name)
        current_image_path = images_dir + file_name
        #print(current_image_path)
        # img = cv2.imread(current_image_path)
        img = ndimage.imread(current_image_path)
        images.append(img)
        img_flipped = np.fliplr(img)
        images.append(img_flipped)
        steering_measurement = float(line[3])
        steering_measurements.append(steering_measurement)
        steering_measurements.append(-steering_measurement)

    train_x = np.array(images)
    train_y = np.array(steering_measurements)

    # Gray scale
    if image_depth == 1:
        print("Convert to gray scale ...")
        train_x_gray = np.sum(train_x / 3, axis=3, keepdims=True)

        # print(train_x.shape)
        # print(len(images))
        # exit(0)
        # Normalize
        if norm_image:
            print("Normalization ...")
            train_x_normalized = (train_x_gray/255) - 0.5
        else:
            train_x_normalized = train_x_gray
    else:
        train_x_gray = None
        if norm_image:
            print("Normalization ...")
            train_x_normalized = (train_x/255) - 0.5
        else:
            train_x_normalized = train_x


    # Clip images
    if clip_image:
        print("Data Clipping ...")
        train_x_cliped = train_x_normalized[:, removed_pixels:, :, :]
        image_hight -= removed_pixels
    else:
        train_x_cliped = train_x_normalized
        # print(train_x.shape)
    if save:
        dist_pickle = dict()
        #dist_pickle["images"] = images
        #dist_pickle["train_x_gray"] = train_x_gray
        #dist_pickle["train_x_normalized"] = train_x_normalized
        dist_pickle["train_x_cliped"] = train_x_cliped
        dist_pickle["train_y"] = train_y
        save_nn(dist_pickle, data_path)
    return train_x, train_x_gray, train_x_normalized, train_x_cliped, train_y


def load_images(*args, **kwargs):
    file_path = data_path + "data.p"
    if os.path.isfile(file_path):
        #print("Loading file : %s" %file_path)
        return loc_load_nn_file(file_path)
    else:
        return csv_load_images(*args, **kwargs)
