import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K

training_data = np.load("combinedData25Random.npy")
training_data = training_data.T
kernel_init = tf.keras.initializers.RandomNormal(stddev=0.001)
print(training_data.shape)

def regressor():
    # function to create actor network
        # obs_size - the input shape for the actor network, the same as the observation shape returned by the environment
        # action_size - the output shape of the actor, the same as the action space for the iirabm environment
    # returns the actor model
    num_hidden1 = 400
    num_hidden2 = 300
    num_hidden3 = 200
    input = tf.keras.layers.Input(shape=(11,))

    hidden = layers.Dense(num_hidden1, activation="relu", kernel_initializer=kernel_init)(input)
    hidden = layers.Dense(num_hidden2, activation="relu",kernel_initializer=kernel_init)(hidden)
    hidden = layers.Dense(num_hidden3, activation="relu",kernel_initializer=kernel_init)(hidden)

    output = layers.Dense(1, activation='relu',kernel_initializer=kernel_init)(hidden)

    model = tf.keras.Model(input, output)

    model.compile(optimizer="Adam", loss="mse")
    return model

cytokines = training_data[:,1:]
oxydef = training_data[:,0]

regressor = regressor()

regressor.fit(cytokines, oxydef, epochs=100, verbose=1, validation_split=.1, batch_size=1000)

regressor.save("oxydef_regressor.h5")
