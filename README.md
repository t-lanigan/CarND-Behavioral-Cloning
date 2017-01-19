# Behavioral Cloning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, a trained car drives in a simulated environment by cloning the behavior as seen during training mode.  Leveraging TensorFlow and Keras, a deep learning network predicts the proper steering angle given training examples.

[Demo Video](https://www.youtube.com/embed/juPqoEllio8)

## Dependencies

Install Python Dependencies with Anaconda (conda install …)
* numpy
* flask-socketio
* eventlet
* pillow
* h5py

Install Python Dependencies with pip (pip install ...)
* keras

## Files
* `model.py` - The script used to create and train the model.
* `drive.py` - The script to drive the car.
* `model.json` - The model architecture.
* `model.h5` - The model weights.

## Udacity Simulator

Udacity created a simulator based on the Unity engine that uses real game physics to create a close approximation to real driving.

### Download

* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
* [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
* [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

## Network

### Approach

Given that we are trying to map raw pixel data to a steering angle, the first thing to note when considering an architecture is that this is a regression problem, and therefore the final layer must output a continous value.

I considered using transfer learning using ImageNet weights along with a network such as VGG, ResNet, or GoogLeNet, however I consider images contained in ImageNet to be a lot different than those from the Udacity simulator.  If leveraging transfer learning, then perhaps extracting earlier layers could be useful where more specific features have not been learned.  Even though ImageNet is built on a classfication problem, since we can remove the top layers, we can still replace the final layer to output a single continuous value.

Next, I took a look at the solution documented in the [NVIDIA Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), in which raw pixels are mapped steering commands.  This is remarkedly similar to our given problem.  Because of the similarity I decided it would be a good starting point.  The Nvidia architecture is small compared to the previously considered architectures with only 9 layers.  After experimenting with a rough replication of the network, I found that I could train relatively fast, and because of this, I decided that did not need transfer learning to complete this project, opting to stick with the simpler Nvidia network.

One decision I made in designing the network was around code reuse.  I decided that all image preprocessing belongs in the pipeline itself as apposed to a separate process external to the pipeline.  With all image preprocessing in the pipeline, we are no longer required to modify `drive.py` with any modifications we make to image preprocessing.

After getting the initial network running, I experimented with different dropout layers and activation functions.  

For activations, I read a [Paper on ELU Activations](https://arxiv.org/pdf/1511.07289v1.pdf), which led me to experiment, comparing the training time and loss for RELU vs ELU activations.  After several trials I concluded that ELUs did indeed give marginally faster performance and lower loss.  ELU activations offer the same protection against vanishing gradiant as RELU, and in addition, ELUs have negative values, which allows them to push the mean activations closer to zero, improving the efficiency of gradient descent.

For dropout, I ran trials with values between 0.2 and 0.5 for fraction of inputs to drop, as well as which layers to include a dropout operation.  I found that my model performed poorly in autonomous mode when including dropout layers in the final fully connected layers.  My intuition here is that dropout may not be appropriate for every layer in regression problems.  In classification problems we are only concerned softmax probabilities relative to another class, so even if dropout effects the final value, it should not matter because we only care about the value relative to other classes.  With regression, we care about the final value, so dropout might have negative effects.  To avoid this dilemma, I chose l2 regularization in the fully connected layers.  Initially, this prevented the model from producing sharp turns, but was fixed after reducing the weight penalty.

### Architecture

My architecture is modeled after the network depicted in [NVIDIA Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  The architecture is a traditional feed-foward layered architecture in which the output of one layer is fed to the layer above.  At a high level the network consists of preprocessing layers, 5 convolutional layers, followed by 3 fully connected layers, and a final output layer.  Since we are working with a regression problem, the output layer is a single continuous value, as apposed to the softmax probabilities used for classification tasks such as traffic sign identification.

Before the first convolutional layer, a small amount of preprocessing takes place within the pipeline.  This includes cropping the image, resizing, and batch normalization.

Each convolitional has a 1x1 stride, and uses a 2x2 max pooling operation to reduce spatial resolution. The first three convolutional layers use a 5x5 filter, while the final two use a 3x3 filter as the input dimensionality is reduced.

For regularization, a spatial dropout operation is added after each convolutional layer.  Spatial dropout layers drop entire 2D features maps instead of individual features.

For non-linearity, ELU activationd are used for each convolutional, as well as each fully connected layer.

The output from the forth convolutional layer is flattened and fed into a regressor composed of four fully connected layers.  The fully connected layers each reduce the number of features with the final layer outputting a single continuous value.  As noted above, l2 regularization is leveraged in the fully connected layers.

See the diagram below.  This diagram is modified from the original source diagram found in the the [NVIDIA Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).  The values have been modified to represent input sizes of our recorded training data and to include the additional preprocessing layers.

![Architecture](./images/architecture.png)

## Data Collection

In order for the car to drive in autonmous mode, we must first record training data and then train our model using the recorded data.

![Data Collection through Udacity Simulator](./images/data-collection.png)

We start by recording while driving normally around the track a few times.  However, it is also helpful to record recovery data, where recording starts when you are turning away from the edge back to the center.

### Sample Images

![Sample Images](./images/sample-images.png)

## Training

A trained model is able to predict a steering angle given a camera image.  But, before sending our recorded images and steering angle data into the network for training, we can improve performance by limiting how much data is stored in memory as well as image preprocessing.

### Image Generator

The entire set of images used for training would consume a large amount of memory.  A python generator is leveraged so that only a single batch is contained in memory at a time.

### Image Preprocessing

Image preprocessing is contained within the network pipeline.  This allows reuse so that no additional modifications are required within `drive.py`.

First the image is cropped above the horizon to reduce the amount of information the network is required to learn.  Next the image is resized to further reduce required processing.  Finally normalization is applied to each mini-batch.  This helps keep weight values small, improving numerical stability. In addition since our mean is relatively close to zero, the gradient descent optimization will have less searching to do when minimizing loss.

![Image Preprocessing](./images/preprocess.png)

### Network Output

Once the network is trained, the model definition as well as the trained weights are saved so that the autonomous driving server can reconstruct the network and make predictions given the live camera stream.

Now we can run the simulator in autonomous mode and start the driver server.

```
python drive.py model.json
```

The autonomous driving server sends predicted steering angles to the car using the trained network.  Here we can test how well the model performs.  If the car makes mistakes, we return to training mode to collect more training data.

