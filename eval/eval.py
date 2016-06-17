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
   with open(result_file) as rf:
      for line in rf:
         line = line.rstrip().split("|")
         print line
         exit()
         real_image = line[0]
         gen_image  = line[1]
        
         print real_image
         exit() 



