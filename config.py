import os
from keras import backend as K

K.set_image_data_format('channels_last')

# general
data_dir = 'data/*'
test_dir = 'data/test/*'
label_deir= 'data/label.txt'
val_dir = 'data/val/*'
checkpoint_dir = 'checkpoints/'
validation_dir = 'data/validation/'
prediction_dir = 'data/prediction/'

directory = os.path.dirname(checkpoint_dir)
if not os.path.exists(directory):
    os.makedirs(directory)
    
directory = os.path.dirname(validation_dir)
if not os.path.exists(directory):
    os.makedirs(directory)
    
directory = os.path.dirname(prediction_dir)
if not os.path.exists(directory):
    os.makedirs(directory)

# bdsscgan
input = 4
output = 4
size = 240
epochs = 110
sequence_length = 10
kernel_depth = 24

checkpoint_gen_name = checkpoint_dir + 'gen.hdf5'
checkpoint_disc_name = checkpoint_dir + 'disc.hdf5'
