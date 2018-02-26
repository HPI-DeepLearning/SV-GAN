from keras.optimizers import Adam

from model import *
from utils import *
from config import *

opt = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  
gen = Generator((sequence_length, size, size, input), output, kernel_depth, size*size*sequence_length)
gen.compile(loss='mae', optimizer=opt)
gen.load_weights(checkpoint_gen_name)

# List sequences  
sequences = prepare_data(test_dir)

progbar = keras.utils.Progbar(len(sequences))

for s in range(len(sequences)):
    
    progbar.add(1)
    sequence = sequences[s]
    x, y = load(sequence, sequence_length)
    
    for i in range(len(x)):
    
        # predict
        generated_y = gen.predict(x[i])
        save_image(x[i] / 2 + 0.5, y[i], re_shape(generated_y), prediction_dir + "te{}.png".format(s))
