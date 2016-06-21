# Convolutional Neural Network Autoencoder

This is an attempt to create an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
using a convolutional neural network. The network consists of an initial dropout layer and
three convolutional layers. Three more fully connected layers bring down the image to a
dimension of just 256, where it is then built back up through three more fully connected layers,
and finally three deconvolution layers using the `conv2d_transpose` function in Tensorflow.

## Architecture
A model of the graph using `Tensorboard` can be seen below.

![Graph](https://raw.githubusercontent.com/cameronfabbri/Autoencoder/master/graph.png)

## Training
Training was done on the ![Microsoft COCO Dataset](http://mscoco.org/) which included around
82,000 images.

## Testing

Tests were done on a holdout set of the COCO dataset that held about 40,000 images. A few
results at different training steps can be seen below. The image on the left is the image the
system is trying to create from the 256 dimensional vector it is given, and the image on the
right is what it actually creates.


### 1,000 Training Iterations
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_1000/image-0.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_1000/image-10.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_1000/image-12.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_1000/image-14.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_1000/image-3.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_1000/image-5.png?raw=true)


### 5,000 Training Iterations
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_5000/image-0.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_5000/image-10.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_5000/image-12.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_5000/image-14.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_5000/image-3.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_5000/image-5.png?raw=true)


### About 390,000 Training Iterations
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_390000/image-0.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_390000/image-10.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_390000/image-12.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_390000/image-14.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_390000/image-3.png?raw=true)
![im](https://github.com/cameronfabbri/Autoencoder/blob/master/test_results/step_390000/image-5.png?raw=true)


Clearly there isn't much of a change between 5,000 iterations and 390,000 iterations.
