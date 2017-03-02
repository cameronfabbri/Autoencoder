## MNIST Autoencoder

Just an autoencoder example, specifically for the MNIST dataset.
This compresses the image from 786 dimmensions down to 8.

## Usage
```python
>>> python autoencoder.py
```

This will automatically download the MNIST dataset. Batch size is set to 1000 so you might want to change it depending on your system.
A checkpoint is included in the `checkpoint` folder, and will automatically be loaded upon running. To train your own, delete the checkpoint.

### Results

It didn't do too bad considering the images are compressed to only 8 dimensions. Below are some examples.

![img](http://i.imgur.com/Qa6HfhT.png)


