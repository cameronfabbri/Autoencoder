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
test results can be seen below after about 20,000 iterations. The original image is the
image the system is trying to create from the 256 dimensional vector it is given. The system
has not seen these images before.

| Original image | Generated Image |
|----------------|:---------------:|
|![im](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/im-0.png?raw=true)|![gen](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/gen-0.png?raw=true)|
|![im](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/im-1.png?raw=true)|![gen](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/gen-1.png?raw=true)|
|![im](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/im-2.png?raw=true)|![gen](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/gen-2.png?raw=true)|
|![im](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/im-3.png?raw=true)|![gen](https://github.com/cameronfabbri/Autoencoder/blob/master/evaluations/images/gen-3.png?raw=true)|


