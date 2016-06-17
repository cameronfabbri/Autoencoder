# directory of your root data, must not end with a /
data_dir = "/home/fabbric/data_dir"
#data_dir = "/mnt/data3/data_dir"

# specific dataset. assumes data_dir/dataset/images
dataset  = "birds_2011"

#
train_perc = 1
test_perc  = 0

# model checkpoint path
checkpoint_dir = "../models/"

# directory to store evaluation logs
eval_dir = "../evaluations/"

batch_size = 30

# the number of classes in the dataset. This will be the amount of nodes
# in the last layer
num_classes = 200
