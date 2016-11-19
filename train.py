import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
import cv2
import time
import requests
from tqdm import tqdm
import gzip
import cPickle as pickle

def train(mnist, batch_size):
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False)
      epoch_num   = tf.Variable(0, trainable=False) # name=

      images = tf.placeholder(tf.float32, [batch_size, 784])

      logits = architecture.inference(images)
      loss   = architecture.loss(images, logits)
      
      train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

      variables = tf.all_variables()

      init = tf.initialize_all_variables()
      sess = tf.Session()

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()
      
      sess.run(init)

      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.train.SummaryWriter(checkpoint_dir+"training", graph_def=graph_def)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())
      
      tf.train.start_queue_runners(sess=sess)

      #for step in xrange(10000):
      step = 0
      while True:
         step += 1
         _, loss_value, generated_image, imgs = sess.run([train_op, loss, logits, images])
         #print logs
         print "Step: " + str(step) + " Loss: " + str(loss_value)

         # save for tensorboard
         if step%5000 == 0 and step != 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

            print "Saving model..."
            saver.save(sess, checkpoint_dir+"training", global_step=step)
            

def main(argv=None):
   url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'

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
  
   train = train_set[0]
   val   = val_set[0]
   test  = test_set[0]

   a = np.concatenate((train, val), axis=0)
   mnist = np.concatenate((a, test), axis=0)
   train(mnist)


if __name__ == "__main__":
   tf.app.run()

