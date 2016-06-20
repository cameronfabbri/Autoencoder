# Convolutional Neural Network Autoencoder

This is an attempt to create an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
using a convolutional neural network. The network consists of an initial dropout layer and
three convolutional layers. Three more fully connected layers bring down the image to a
dimension of just 256, where it is then built back up through three more fully connected layers,
and finally three deconvolution layers using the `conv2d_transpose` function in Tensorflow.

## Architecture
A model of the graph using `Tensorboard` can be seen below.

![Graph](https://raw.githubusercontent.com/cameronfabbri/Autoencoder/master/graph.png)

## Testing

Coming soon..
