import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
import numpy as np
import requests
import random
import gzip
import os

batch_size = 256

'''
   Kullback Leibler divergence
   https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
   https://github.com/fastforwardlabs/vae-tf/blob/master/vae.py#L178
'''
def kullbackleibler(mu, log_sigma):
   return -0.5*tf.reduce_sum(1+2*log_sigma-mu**2-tf.exp(2*log_sigma),1)


'''
   Leaky RELU
'''
def lrelu(x, leak=0.2, name="lrelu"):
   return tf.maximum(x, leak*x)


def encoder(x):
   
   e_conv1 = tf.layers.conv2d(x, 16, 5, strides=2,   name='e_conv1')
   e_conv1 = lrelu(e_conv1) 
   print 'conv1: ', e_conv1

   e_conv2 = tf.layers.conv2d(e_conv1, 32, 5, strides=2,   name='e_conv2')
   e_conv2 = lrelu(e_conv2)
   print 'conv2: ', e_conv2
   
   e_conv2_flat = tf.reshape(e_conv2, [batch_size, -1])

   '''
      z is distributed as a multivariate normal with mean z_mean and diagonal
      covariance values sigma^2

      z ~ N(z_mean, np.exp(z_log_sigma)**2)
   '''
   z_mean = tf.layers.dense(e_conv2_flat, 32, name='mean')
   z_log_sigma = tf.layers.dense(e_conv2_flat, 32, name='stddev')
   

   return z_mean, z_log_sigma

def decoder(z):
   print
   print 'z: ', z

   d_fc1 = tf.layers.dense(z, 7*7*32, name='d_fc1')
   d_fc1 = lrelu(d_fc1)
   print 'd_fc1: ', d_fc1
   d_fc1 = tf.reshape(d_fc1, [batch_size, 7,7,32])

   e_transpose_conv1 = tf.layers.conv2d_transpose(d_fc1, 16, 5, strides=2, name='e_transpose_conv1')
   e_transpose_conv1 = lrelu(e_transpose_conv1)
   print 'e_transpose_conv1: ', e_transpose_conv1

   e_transpose_conv2 = tf.layers.conv2d_transpose(e_transpose_conv1, 1, 5, strides=2, name='e_transpose_conv2')
   e_transpose_conv2 = tf.nn.sigmoid(e_transpose_conv2)
   e_transpose_conv2 = e_transpose_conv2[:,:28,:28,:]
   print 'e_transpose_conv2: ', e_transpose_conv2

   return e_transpose_conv2


def train(mnist_train, mnist_test):
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False, name='global_step')

      # placeholder for mnist images
      images      = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])

      # encode images to 8 dim vector
      z_mean, z_log_sigma = encoder(images)

      # reparameterization trick
      epsilon = tf.random_normal(tf.shape(z_log_sigma), name='epsilon')
      z = z_mean + epsilon * tf.exp(z_log_sigma) # N(mu, sigma**2)

      decoded = decoder(z)
      
      #reconstructed_loss = -tf.reduce_sum(images*tf.log(1e-10 + decoded) + (1-images)*tf.log(1e-10+1-decoded), 1)
      reconstructed_loss = tf.nn.l2_loss(images-decoded)
      latent_loss = kullbackleibler(z_mean, z_log_sigma)

      cost = tf.reduce_mean(reconstructed_loss+latent_loss)

      train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())

      init = tf.initialize_all_variables()
      sess = tf.Session()
      sess.run(init)

      try: os.makedirs('checkpoint/images/')
      except: pass

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
         _, loss_ = sess.run([train_op, cost], feed_dict={images: batch_images})
         loss_ = sess.run([cost], feed_dict={images:batch_images})[0]
         print 'Step: ' + str(step) + ' Loss: ' + str(loss_)

         if step%500 == 0:
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
               plt.imsave('checkpoint/images/'+str(step)+'_'+str(c)+'real.png', real)
               plt.imsave('checkpoint/images/'+str(step)+'_'+str(c)+'dec.png', dec)
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
