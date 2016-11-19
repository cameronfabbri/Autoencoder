import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
import numpy as np
import requests
import gzip
import os

# import the functions we are going to use
from tensorflow_ops import _conv_layer, _transpose_conv_layer, _fc_layer, lrelu

batch_size = 10

def encoder(x):
   # convolutional layer with a leaky Relu activation
   e_conv1 = lrelu(_conv_layer(x, 2, 2, 32, 'e_conv1'))
   print 'conv1: ', e_conv1
   print

   # convolutional layer with a leaky Relu activation
   e_conv2 = lrelu(_conv_layer(e_conv1, 2, 2, 64, 'e_conv2'))
   print 'conv2: ', e_conv2
   print
   
   # convolutional layer with a leaky Relu activation
   e_conv3 = lrelu(_conv_layer(e_conv2, 2, 2, 32, 'e_conv3'))
   print 'conv3: ', e_conv3
   print
  
   # fully connected layer with a leaky Relu activation
   # The 'True' here means that we are flattening the input
   e_fc1 = lrelu(_fc_layer(e_conv2, 512, True, 'e_fc1'))
   print 'fc1: ', e_fc1
   print

   # fully connected layer with a leaky Relu activation
   # the output from the previous fully connected layer is
   # already flat, so no need to flatten, hence 'False'
   e_fc2 = lrelu(_fc_layer(e_fc1, 256, False, 'e_fc2'))
   print 'fc2: ', e_fc2
   print

   return e_fc2

def decoder(x):
   print 'x: ', x
   print
  
   
   # fully connected layer with a leaky Relu activation
   d_fc1 = lrelu(_fc_layer(x, 512, False, 'd_fc1'))

   # reshape for use in transpose convolution (deconvolution) 
   # must match conv layers in encoder
   d_fc1 = tf.reshape(d_fc1, (batch_size, 4, 4, 32))
   print 'd_fc1: ', d_fc1
   print
 
   # transpose convolution with a leaky relu activation
   e_transpose_conv1 = lrelu(_transpose_conv_layer(d_fc1, 2, 2, 32, 'e_transpose_conv1'))
   print 'e_transpose_conv1: ', e_transpose_conv1
   print

   # transpose convolution with a leaky relu activation
   e_transpose_conv2 = lrelu(_transpose_conv_layer(e_transpose_conv1, 2, 2, 64, 'e_transpose_conv2'))
   print 'e_transpose_conv2: ', e_transpose_conv2
   print
   
   # transpose convolution with a leaky relu activation
   e_transpose_conv3 = lrelu(_transpose_conv_layer(e_transpose_conv2, 2, 2, 1, 'e_transpose_conv3'))
   print 'e_transpose_conv3: ', e_transpose_conv3
   print

   # since transpose convs make the resolution go 4->8->16->32 (because stride 2)
   # we need to crop to original mnist size (28,28)
   e_transpose_conv3 = e_transpose_conv3[:,:28,:28,:]
   return e_transpose_conv3


def train(mnist_train, mnist_test):
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False, name='global_step')

      # placeholder for mnist images
      images      = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])

      # encode images to 256 dim vector
      encoded = encoder(images)

      # decode 256 dim vector to (28,28) dim image
      decoded = decoder(encoded)

      # loss is the squared error between the true image and the one we
      # decoded from the 256 dim vector
      loss = tf.reduce_mean(tf.pow(images - decoded, 2))
     
      train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

      init = tf.initialize_all_variables()
      sess = tf.Session()
      sess.run(init)

      step = 0
      while True:
         step += 1

         # get random images from the training set
         batch_images = random.sample(mnist_train, batch_size)

         # send through the network
         _, loss_ = sess.run([train_op, loss], feed_dict={images: batch_images})
         print 'Step: ' + str(step) + ' Loss: ' + str(loss_)

         if step%1000 == 0:

            # get random images from the test set
            batch_images = random.sample(mnist_test, batch_size)

            # encode them using the encoder, then decode them
            encode_decode = sess.run(decoded, feed_dict={images: batch_images})

            # write out to files
            for img in encode_decode:
               img = np.squeeze(img)
               plt.imsave('images/image_'+str(step)+'.png', img)


def main(argv=None):
   # mnist data in gz format
   url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

   # check if it's already downloaded
   if not os.path.isfile('./mnist.pkl.gz'):
      print 'Downloading mnist...'
      with open('mnist.pkl.gz', 'wb') as f:
         r = requests.get(url)
         if r.status_code == 200:
            f.write(r.content)
         else:
            print 'Could not connect to ', url

   f = gzip.open('mnist.pkl.gz', 'rb')
   train_set, val_set, test_set = pickle.load(f)

   mnist_train = []
   mnist_test = []

   # reshape mnist to make it easier for understanding convs
   for t,l in zip(*train_set):
      mnist_train.append(np.reshape(t, (28,28,1)))
   for t,l in zip(*val_set):
      mnist_train.append(np.reshape(t, (28,28,1)))
   for t,l in zip(*test_set):
      mnist_test.append(np.reshape(t, (28,28,1)))

   mnist_train = np.asarray(mnist_train)
   mnist_test  = np.asarray(mnist_test)

   train(mnist_train, mnist_test)

if __name__ == '__main__':
   tf.app.run()
