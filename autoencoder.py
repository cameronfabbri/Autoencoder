import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
import numpy as np
import requests
import random
import gzip
import os

batch_size = 1000

'''
   Leaky RELU
'''
def lrelu(x, leak=0.2, name="lrelu"):
   return tf.maximum(x, leak*x)


def encoder(x):
   
   e_conv1 = slim.convolution(x, 32, 2, stride=2, activation_fn=tf.identity, normalizer_fn=slim.batch_norm, scope='e_conv1')
   e_conv1 = lrelu(e_conv1) 
   print 'conv1: ', e_conv1

   e_conv2 = slim.convolution(e_conv1, 64, 2, stride=2, activation_fn=tf.identity, normalizer_fn=slim.batch_norm, scope='e_conv2')
   e_conv2 = lrelu(e_conv2)
   print 'conv2: ', e_conv2
   
   # convolutional layer with a leaky Relu activation
   e_conv3 = slim.convolution(e_conv2, 128, 2, stride=2, activation_fn=tf.identity, normalizer_fn=slim.batch_norm, scope='e_conv3')
   e_conv3 = lrelu(e_conv3)
   print 'conv3: ', e_conv3
  
   e_conv3_flat = tf.reshape(e_conv3, [batch_size, -1])

   e_fc1 = slim.fully_connected(e_conv3_flat, 256, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc1')
   e_fc1 = lrelu(e_fc1)
   print 'fc1: ', e_fc1
   
   e_fc2 = slim.fully_connected(e_fc1, 64, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc2')
   e_fc2 = lrelu(e_fc2)
   print 'fc2: ', e_fc2
   
   e_fc3 = slim.fully_connected(e_fc2, 32, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc3')
   e_fc3 = lrelu(e_fc3)
   print 'fc3: ', e_fc3
   
   e_fc4 = slim.fully_connected(e_fc3, 8, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc4')
   e_fc4 = lrelu(e_fc4)
   print 'fc4: ', e_fc4
   return e_fc4

def decoder(x):
   print
   print 'x: ', x

   d_fc1 = slim.fully_connected(x, 32, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc1')
   d_fc1 = lrelu(d_fc1)
   print 'd_fc1: ', d_fc1

   d_fc2 = slim.fully_connected(x, 64, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc2')
   d_fc2 = lrelu(d_fc2)
   print 'd_fc2: ', d_fc2

   d_fc3 = slim.fully_connected(x, 256, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc3')
   d_fc3 = lrelu(d_fc3)
   print 'd_fc3: ', d_fc3

   d_fc3 = tf.reshape(d_fc3, [batch_size, 4, 4, 16])
   print 'd_fc3: ', d_fc3
 
   e_transpose_conv1 = slim.convolution2d_transpose(d_fc3, 64, 2, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_transpose_conv1')
   e_transpose_conv1 = lrelu(e_transpose_conv1)
   print 'e_transpose_conv1: ', e_transpose_conv1

   e_transpose_conv2 = slim.convolution2d_transpose(e_transpose_conv1, 32, 2, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_transpose_conv2')
   e_transpose_conv2 = lrelu(e_transpose_conv2)
   print 'e_transpose_conv2: ', e_transpose_conv2

   e_transpose_conv3 = slim.convolution2d_transpose(e_transpose_conv2, 1, 2, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_transpose_conv3')
   e_transpose_conv3 = lrelu(e_transpose_conv3)
   e_transpose_conv3 = e_transpose_conv3[:,:28,:28,:]
   print 'e_transpose_conv3: ', e_transpose_conv3
   return e_transpose_conv3


def train(mnist_train, mnist_test):
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False, name='global_step')

      # placeholder for mnist images
      images      = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])

      # encode images to 128 dim vector
      encoded = encoder(images)

      # decode 128 dim vector to (28,28) dim image
      decoded = decoder(encoded)

      loss = tf.nn.l2_loss(images - decoded)
     
      train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())

      init = tf.initialize_all_variables()
      sess = tf.Session()
      sess.run(init)

      try:
         os.mkdir('images/')
      except:
         pass
      try:
         os.mkdir('checkpoint/')
      except:
         pass

      ckpt = tf.train.get_checkpoint_state('checkpoint/')
      if ckpt and ckpt.model_checkpoint_path:
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Model restored'
         except:
            print 'Could not restore model'
            pass

      step = 0
      while True:
         step += 1

         # get random images from the training set
         batch_images = random.sample(mnist_train, batch_size)

         # send through the network
         _, loss_ = sess.run([train_op, loss], feed_dict={images: batch_images})
         print 'Step: ' + str(step) + ' Loss: ' + str(loss_)

         if step%100 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, "checkpoint/checkpoint", global_step=global_step)

            # get random images from the test set
            batch_images = random.sample(mnist_test, batch_size)

            # encode them using the encoder, then decode them
            encode_decode = sess.run(decoded, feed_dict={images: batch_images})

            # write out a few
            c = 0
            for real, dec in zip(batch_images, encode_decode):
               dec, real = np.squeeze(dec), np.squeeze(real)
               plt.imsave('images/'+str(step)+'_'+str(c)+'real.png', real)
               plt.imsave('images/'+str(step)+'_'+str(c)+'dec.png', dec)
               if c == 5:
                  break
               c+=1

def main(argv=None):
   # mnist data in gz format
   url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

   # check if it's already downloaded
   if not os.path.isfile('mnist.pkl.gz'):
      print 'Downloading mnist...'
      with open('mnist.pkl.gz', 'wb') as f:
         r = requests.get(url)
         if r.status_code == 200:
            f.write(r.content)
         else:
            print 'Could not connect to ', url

   print 'opening mnist'
   f = gzip.open('mnist.pkl.gz', 'rb')
   train_set, val_set, test_set = pickle.load(f)

   mnist_train = []
   mnist_test = []

   print 'Reading mnist...'
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
