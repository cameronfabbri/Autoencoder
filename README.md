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

The images on the left are the true images, and on the right are after encoding and decoding. This was run on the MNIST test set.

![realimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_0real.png?raw=true)![decimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_0dec.png?raw=true)


![realimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_1real.png?raw=true)![decimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_1dec.png?raw=true)


![realimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_2real.png?raw=true)![decimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_2dec.png?raw=true)


![realimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_3real.png?raw=true)![decimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_3dec.png?raw=true)


![realimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_4real.png?raw=true)![decimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_4dec.png?raw=true)


![realimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_5real.png?raw=true)![decimg1](https://github.com/cameronfabbri/Autoencoder/blob/master/images/22900_5dec.png?raw=true)





