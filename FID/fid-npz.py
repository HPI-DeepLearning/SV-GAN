#!/usr/bin/env python3

import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

########
# PATHS
########
data_path = 'data/GT/ET/' # set path to training set images
output_path = 'data/GT/fid_ET.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(data_path, '*.png'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

images = images.reshape(images.shape[0], 240, 240, 1) #1 color channel:
images = np.tile(images,3)

print("%d images found and loaded" % len(images))


print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")
