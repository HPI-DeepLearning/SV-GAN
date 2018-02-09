from keras.models import Model
from keras.layers import Bidirectional, Input, Concatenate, Cropping3D, Dense, Flatten, TimeDistributed, ConvLSTM2D, LeakyReLU
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
import keras.backend as K


def conv_layer(layer, depth, size):
    conv = TimeDistributed(Conv2D(depth, size, padding='same'))(layer)
    conv = LeakyReLU(0.2)(conv)
    return TimeDistributed(BatchNormalization())(conv)

def lstm_layer(layer, depth, size):
    conv = Bidirectional(ConvLSTM2D(depth, size, padding='same', return_sequences=True), merge_mode='sum')(layer)
    return TimeDistributed(BatchNormalization())(conv)
    
def lstm_merge_layer(layer, depth, size):
    conv = Bidirectional(ConvLSTM2D(depth, size, padding='same', return_sequences=False), merge_mode='sum')(layer)
    return TimeDistributed(BatchNormalization())(conv)
    
def Generator(input_shape, output, kernel_depth, pixels, kernel_size=5):
    input = Input(shape=input_shape)
    
    conv_seperable = TimeDistributed(SeparableConv2D(2 * kernel_depth, kernel_size, padding='same'))(input)
    conv_seperable = Activation('tanh')(conv_seperable)
    conv_seperable = TimeDistributed(BatchNormalization())(conv_seperable)

    conv_128 = conv_layer(conv_seperable, 2 * kernel_depth, kernel_size)
    pool_64 = TimeDistributed(MaxPooling2D())(conv_128)

    conv_64 = conv_layer(pool_64, 2 * kernel_depth, kernel_size)
    pool_32 = TimeDistributed(MaxPooling2D())(conv_64)

    conv_32 = conv_layer(pool_32, 4 * kernel_depth, kernel_size)
    pool_16 = TimeDistributed(MaxPooling2D())(conv_32)

    conv_16 = conv_layer(pool_16, 8 * kernel_depth, kernel_size)
    pool_8 = TimeDistributed(MaxPooling2D())(conv_16)

    conv_8 = lstm_layer(pool_8, 8 * kernel_depth, kernel_size)

    up_16 = concatenate([TimeDistributed(UpSampling2D())(conv_8), conv_16])
    up_conv_16 = conv_layer(up_16, 8 * kernel_depth, kernel_size)

    up_32 = concatenate([TimeDistributed(UpSampling2D())(up_conv_16), conv_32])
    up_conv_32 = conv_layer(up_32, 4 * kernel_depth, kernel_size)

    up_64 = concatenate([TimeDistributed(UpSampling2D())(up_conv_32), conv_64])
    up_conv_64 = conv_layer(up_64, 2 * kernel_depth, kernel_size)

    up_128 = concatenate([TimeDistributed(UpSampling2D())(up_conv_64), conv_128])
    up_conv_128 = conv_layer(up_128, kernel_depth, kernel_size)
	
	#adding sv-input&output data
    x0 = lstm_merge_layer(up_conv_128, 8 * kernel_depth, kernel_size)
	x1 = Flatten()(x0)
	age = Input(shape=(1,)) #input_ages
    added = concatenate([x1, age])
    out = keras.layers.Dense(64)(added)
    out = keras.layers.Dense(3)(out)
	survival_rate = Activation('softmax')(out)
	
    final = TimeDistributed(Conv2D(output, 1))(up_conv_128)
    final = Reshape((pixels, output))(final)
    final = Activation('softmax')(final)
	
    model = Model([input, age], [final, survival_rate], name="Generator")
    return model
   
def Discriminator(input_shape, generator_shape, kernel_depth, kernel_size=5):
    real_input = Input(shape=input_shape)
    generator_input = Input(shape=generator_shape)    
    input = Concatenate()([real_input, generator_input])
  
    conv_seperable = TimeDistributed(SeparableConv2D(kernel_depth, kernel_size, padding='same'))(input)
    conv_seperable = Activation('tanh')(conv_seperable)
    conv_seperable = TimeDistributed(BatchNormalization())(conv_seperable)

    conv_128 = conv_layer(conv_seperable, kernel_depth, kernel_size)
    pool_64 = TimeDistributed(MaxPooling2D())(conv_128)

    conv_64 = conv_layer(pool_64, 2 * kernel_depth, kernel_size)
    pool_32 = TimeDistributed(MaxPooling2D())(conv_64)

    conv_32 = conv_layer(pool_32, 2 * kernel_depth, kernel_size)
    pool_16 = TimeDistributed(MaxPooling2D())(conv_32)

    conv_16 = conv_layer(pool_16, 4 * kernel_depth, kernel_size)
    pool_8 = TimeDistributed(MaxPooling2D())(conv_16)

    conv_8 = lstm_layer(pool_8, 4 * kernel_depth, kernel_size)
    
    x = Flatten()(conv_8)
    x = Dense(2, activation="softmax")(x)
    
    model = Model([real_input, generator_input], x, name="Discriminator")
    return model

     
def Combine(gen, disc, input_shape, new_sequence):
    input = Input(shape=input_shape)
    generated_image, survival_rate = gen([input, numpy.ones((1,))])

    reshaped = Reshape(new_sequence)(generated_image)
    
    DCGAN_output = disc([input, reshaped])

    DCGAN = Model(inputs=[input],
                  outputs=[generated_image, DCGAN_output],
                  name="Combined")

    return DCGAN
