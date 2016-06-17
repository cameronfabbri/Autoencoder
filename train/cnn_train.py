import tensorflow as tf
import numpy as np
import os
import sys

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../inputs/')
sys.path.insert(0, '../model/')

import config
import input_
import architecture

checkpoint_dir = config.checkpoint_dir
batch_size = config.batch_size


def train():
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False)

      images = input_.inputs("train", batch_size)
      images = tf.div(images, 255)

      logits = architecture.inference(images, "train")

      # loss is the l2 norm of my input vector (the image) and the output vector
      loss = architecture.loss(images, logits)
      
      train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)

      variables = tf.all_variables()

      init = tf.initialize_all_variables()

      sess = tf.Session()

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()
      
      sess.run(init)
      print "Running session"

      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.train.SummaryWriter(checkpoint_dir+"training", graph_def=graph_def)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())
      
      tf.train.start_queue_runners(sess=sess)

      for step in xrange(10000):
         _, loss_value, logs, imgs = sess.run([train_op, loss, logits, images])
         #print logs
         print imgs
         print "Step: " + str(step) + " Loss: " + str(loss_value)

         # save for tensorboard
         if step%100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

         if step%1000 == 0:
            saver.save(sess, checkpoint_dir+"training", global_step=step)
            

def main(argv=None):
   train()


if __name__ == "__main__":
   tf.app.run()

