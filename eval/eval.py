"""

Cameron Fabbri

Evaluation by just looking at the original image and the resulting image from the network

"""


import cv2
import sys

sys.path.insert(0, '../utils/')
import config

result_file = config.result_file

if __name__ == "__main__":
   with tf.Graph().as_default() as graph:

      images = input_.inputs("test", batch_size, 1)

      #sess = tf.Session()

      variables_to_restore = tf.all_variables()

      saver = tf.train.Saver(variables_to_restore)

      summary_op = tf.werge_all_summaries()

      summary_writer = tf.train.SummaryWriter(eval_dir, graph)

      with tf.Session() as sess:

         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
         saver.restore(sess, ckpt.model_checkpoint_path)

         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
         coord = tf.train.Coordinator()

         try:
            tf.train.start_queue_runners(sess=sess)
            threads = []

            for q in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
               threads.extend(q.create_threads(sess, coord=coord, daemon=True, start=True))

               num_iter = int(math.ceil(num_examples/batch_size))

               _, loss_value, generated_image, imgs = sess.run([train_op, loss, logits, images])



