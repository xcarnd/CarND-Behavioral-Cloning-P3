# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[model_arch]: ./figures/model_architecture.jpg "Model architecture"
[training1]: ./figures/out_of_track.jpg "First training result of NVidia model"
[center_lane_1]: ./figures/center_lane/center_1.jpg "Center lane driving - 1"
[center_lane_2]: ./figures/center_lane/center_2.jpg "Center lane driving - 2"
[center_lane_3]: ./figures/center_lane/center_3.jpg "Center lane driving - 3"
[center_lane_4]: ./figures/center_lane/center_4.jpg "Center lane driving - 4"
[center_lane_5]: ./figures/center_lane/center_5.jpg "Center lane driving - 5"
[recovering_1]: ./figures/recovering/recovering_1.jpg "Recovering - 1"
[recovering_2]: ./figures/recovering/recovering_2.jpg "Recovering - 2"
[recovering_3]: ./figures/recovering/recovering_3.jpg "Recovering - 3"
[recovering_4]: ./figures/recovering/recovering_4.jpg "Recovering - 4"
[recovering_5]: ./figures/recovering/recovering_5.jpg "Recovering - 5"
[data_vis1]: ./figures/data_vis.png "Data visualization"
[data_vis2]: ./figures/data_vis_with_flipping.png "Visulization with flipped data points"
[flipping1]: ./figures/center_normal.jpg "Normal Image"
[flipping2]: ./figures/center_flipped.jpg "Flipped Image"
[loss_curve]: ./figures/Loss_Curve.png "Loss Curve"

[NVidiaArchitechture]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.py is the video recording of my vehicle driving autonomously around the track

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model, to a large extent, reused the model from nvidia, which used in one of their research on deep learning of self-driving cars ([Link][NVidiaArchitechture]). 

The model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths with 24/36/48/64 (model.py lines 109-124). The output from the convolution layers are then flattened (model.py line 119) and piping into 4 fully connected layers (model.py lines 120-123) to get the final ouput.

The model uses ReLU layers to introduce nonlinearity between all the convolution layers and fully connected layers (except the final output layer). Data points are cropped and normalized in the model using a Keras Cropping2D layer and Lambda layer (model.py lines 111-113). 

#### 2. Attempts to reduce overfitting in the model

The model does not suffer from overfitting issues, so no regulariztions are introduced to avoid overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 124).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

Since this is a task about deriving something from images, I thought I'd better start with convolution neural network. Following the instructions from the project pages, I started with LeNet-5 model.

In order to gauge how well the model was working, I split my samples into a training set and a validation set. The resulting training loss and validation loss for the LeNet-5 model were quite near. No overfitting.

I ran the simulator with the model and see how the vehicle performed. At first everything looked OK, the vehicle was driving straight steadily. But when it had to make the first turn, it still drove straight and ran out of the lane.

Then I turned to the model mentioned by Nvidia (which was also mentioned in the project guides). The original article from NVidia did not mentioned they had use any methods to avoid overfitting, so I started without regularizations. 

The result was quite impressive. Both the training loss and validation loss were quite small. Running with the simulator, the vehicle was going steadily until when driving across the sharp turn near the end of track 1, it failed to make it and fell into the lake:

![The first trail for NVidia architecture][training1]

The losses were telling me I did not need to worry about the overfitting issue too much, so the failure of driving across sharp turns might indicated there were insufficient data points for the model to learn about turning about corners, especially the sharp ones.

To handle with this, I collected more data, focusing on turning and also recovering from both sides.

Finally the vehicle was able to drive autonomously around track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 109-123) consisted of a convolution neural network with the following layers and layer sizes:

| Layer      | Description                            |
|:----------:|:--------------------------------------:|
| Input      | 160x320x3 RGB image                    |
| Cropping2D | Cropping input into 70x320x3 RGB image |
| Lambda     | Normalizing input (within [-0.5, 0.5]) |
| Conv2D 5x5 | 2x2 stride, output size: 33x158x24     |
| Conv2D 5x5 | 2x2 stride, output depth: 15x77x36     |
| Conv2D 5x5 | 2x2 stride, output depth: 6x37x48      |
| Conv2D 3x3 | 1x1 stride, output depth: 4x35x64      |
| Conv2D 3x3 | 1x1 stride, output depth: 2x33x64      |
| Flatten    | Flatten into 4224 neurons              |
| Dense      | 4224 neurons to 100 neurons            |
| Dense      | 100 neurons to 50 neurons              |
| Dense      | 50 neurons to 10 neurons               |
| Dense      | 10 neurons to 1 neuron                 |

Here is a visualization of the architecture:

![Model Architecture][model_arch]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track 1 using center lane driving, one forward and one backward. Here are some example images of center lane driving:

![Center lane driving - 1][center_lane_1]
![Center lane driving - 2][center_lane_2]
![Center lane driving - 3][center_lane_3]
![Center lane driving - 4][center_lane_4]
![Center lane driving - 5][center_lane_5]

I then recorded the vehicle recovering from the left side and right sides of the road back to the center so that the vehicle would learn to drive back when it was off the center. These images show what a recovery looks like starting from the right side of the road.

![Recovering - 1][recovering_1]
![Recovering - 2][recovering_2]
![Recovering - 3][recovering_3]
![Recovering - 4][recovering_4]
![Recovering - 5][recovering_5]

I also recorded more data when the vehicle was making turns, in order to offering more data to the model to learn how to turn a corner. This was a process much similar to taking more laps but turning off recording when the vehicle was driving straight, and focusing on making turns smoothly.

To augment the data set, I flipped images and angles. By looking at the visualization of data points I've collected:

![Data visualization][data_vis1]

There were more data points with negative steering angles than those of positive, because track 1 has more left turns than right turns.

Flipping images and angles not only yields double data points but also makes the data less bias towards driving along some specific direction.

Here are images before and after being flipped:

![Image from center camera without flipping][flipping1]
![Image from center camera with flipping][flipping2]

Below's the visualization with the flipped data points:

![Data visualization - with flipped data points][data_vis2]

(The unbalance of the histogram came with the fact that if there were samples lying at the borders of bins, the samples must be contributing to either bin, not both.)

After the collection process, I had gathered 11724 data points, including data points augmented by flipping.

As the Nvidia process suggested, I've also converted the sampled images into YUV format (for this reason I've also changed drive.py to make the autonomous driving process using a YUV format input).

Finally I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the loss curve:

![Loss curve for training 100 epochs][loss_curve]

As the training progressed, validation loss kept oscillating around 0.0007, training loss kept reducing, but pretty slowly. So I picked 15 as the final epochs for training.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

