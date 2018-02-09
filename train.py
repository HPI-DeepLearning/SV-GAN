from keras.optimizers import Adam

from model import *
from utils import *
from config import *

# Create optimizers
opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  
gen = Generator((sequence_length, size, size, input), output, kernel_depth, size*size*sequence_length)
gen.compile(loss='mae', optimizer=opt_discriminator)

disc = Discriminator((sequence_length, size, size, input), (sequence_length, size, size, output), kernel_depth)
disc.trainable = False

svc = SVClassifier((sequence_length, size, size, input), (sequence_length, size, size, output), kernel_depth)
svc.trainable= False

combined = Combine(gen, disc, (sequence_length, size, size, input), (sequence_length, size, size, output))
loss = ['categorical_crossentropy', 'binary_crossentropy']
loss_weights = [10, 1]
combined.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

disc.trainable = True
disc.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

if os.path.isfile(checkpoint_gen_name):
    gen.load_weights(checkpoint_gen_name)
if os.path.isfile(checkpoint_disc_name):
    disc.load_weights(checkpoint_disc_name)

# List sequences  
sequences = prepare_data(data_dir)
validation = prepare_data(val_dir)
label= prepare_data(label_dir)

real_y = np.reshape(np.array([0, 1]), (1, 2))
fake_y = np.reshape(np.array([1, 0]), (1, 2))

log = open("train.log",'w')

for e in range(epochs):
    print("Epoch {}".format(e))
    random.shuffle(sequences)
    
    progbar = keras.utils.Progbar(len(sequences))
    
    for s in range(len(sequences)):
        
        sequence = sequences[s] 
        x, y = load(sequence, sequence_length)
        
        for i in range(len(x)):
        
            # train disc on real
            dr_loss = disc.train_on_batch([x[i], y[i]], real_y)
        
            # gen fake
            fake = gen.predict(x[i])
        
            # train disc on fake
            df_loss = disc.train_on_batch([x[i], re_shape(fake)], fake_y)
        
            # train combined    
            disc.trainable = False
            g_loss = combined.train_on_batch(x[i], [np.reshape(y[i], (1, sequence_length*size*size, output)), real_y])
            disc.trainable = True
            
            log.write(str(e) + ", " + str(s) + ", " + str(dr_loss) + ", " + str(df_loss) + ", " + str(g_loss[0]) + ", " + str(g_loss[1]) + ", " + str(opt_dcgan.get_config()["lr"]) + "\n")
            
        progbar.add(1)
            
    # validation
    sequence = validation[random.randrange(0,len(validation))]
    x, y = load(sequence, sequence_length)
        
    for i in range(len(x)):
        random_index = random.randrange(0,len(x))
        generated_y = gen.predict(x[random_index])
        save_image(x[random_index] / 2 + 0.5, y[random_index], re_shape(generated_y), validation_dir + "e_{}.png".format(e))
        
    # save weights
    gen.save_weights(checkpoint_gen_name, overwrite=True)
    disc.save_weights(checkpoint_disc_name, overwrite=True)
