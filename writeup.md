# **Behavioral Cloning** 

## Writeup 

### Presented by Medhat HUSSAIN.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./VN_model.png "Model Visualization"
[image2]: ./data/str_center_2019_01_21_20_11_40_056.jpg "Grayscaling"
[image3]: ./data/center_2019_01_22_20_25_55_270.jpg "Recovery Image"
[image4]: ./data/center_2019_01_22_20_26_50_689.jpg "Recovery Image"
[image5]: ./data/center_2019_01_22_20_27_27_801.jpg "Recovery Image"
[image51]: ./data/center_2019_01_22_20_27_28_135.jpg "Recovery Image"
[image52]: ./data/center_2019_01_22_20_27_29_698.jpg "Recovery Image"
[image53]: ./data/center_2019_01_22_20_27_29_698_flipped.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_training.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* VN_model containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py VN_model "run1"
```
 or simply invoke the batch file "drive.bat"

#### 3. Submission code is usable and readable

The model_training.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 16 and 128 (model_training.py lines 101-182) 

The model includes RELU layers to introduce nonlinearity (code line 116), and the data is normalized in the model using a Keras lambda layer (code line 115). 

#### 2. Attempts to reduce overfitting in the model

The dropout layers was used in order to reduce overfitting (model_training.py lines 171), but by the end it was removed to speedup the learning rate.
To avoid the overfitting I added more FC layers,and that help to increase the generalization. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model_training.py line 154).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make something close to the VGG (but with lower parameters) where we added a group of 
convolutional layers with a pooling layer every two layers and by the end a group of fully connected layers.

At the beginning I tried to make a transfer learning from the (VGG16 and Googlenet), with unfrozen parameters (about 25 Mio par.) but it used to take too
much time for the training and the results was not that better.

I was suffering from the underfitting, both accuracies where oscillating around the 50% and when I try to run the simulator
The car used to have a frozen steering angle which causing by the end to get out of the road. By investigating I figured out that 
the data recorded was focusing mainly on the road center and did not including scenarios to move a way from the lines.  
and here I added such new data which caused a dramatic improvement in the behavior.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model_training.py lines 101-182) consisted of a convolution neural network with the following layers and layer sizes ...

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
lambda_1 (Lambda)            (None, 65, 320, 3)        0         
conv2d_1 (Conv2D)            (None, 65, 320, 128)      9728      
max_pooling2d_1 (MaxPooling2 (None, 16, 80, 128)       0         
conv2d_2 (Conv2D)            (None, 16, 80, 64)        204864    
max_pooling2d_2 (MaxPooling2 (None, 5, 26, 64)         0         
conv2d_3 (Conv2D)            (None, 5, 26, 48)         76848     
conv2d_4 (Conv2D)            (None, 5, 26, 32)         13856     
max_pooling2d_3 (MaxPooling2 (None, 2, 13, 32)         0         
conv2d_5 (Conv2D)            (None, 2, 13, 24)         6936      
conv2d_6 (Conv2D)            (None, 2, 13, 16)         3472      
flatten_1 (Flatten)          (None, 416)               0         
dense_1 (Dense)              (None, 512)               213504    
dense_2 (Dense)              (None, 256)               131328    
dense_3 (Dense)              (None, 128)               32896     
dense_4 (Dense)              (None, 64)                8256      
dense_5 (Dense)              (None, 1)                 65        

Total params: 701,753
_________________________________________________________________

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover and returning back to the center These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image51]
![alt text][image52]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images thinking that this would generalize the model... For example, here is an image that has then been flipped:

![alt text][image52]
![alt text][image53]


After the collection process, I had 40,000 number of data points. I then preprocessed this data by.

- Cropping
- Normalization


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
