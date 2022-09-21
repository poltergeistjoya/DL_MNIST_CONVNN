#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pandas as pd

from absl import flags

from dataclasses import dataclass, field, InitVar
from tensorflow.keras import layers, models, regularizers

# classify MNIST digits with convolutional neural network
# objective 95.5% accuracy

#split train data into train and val

#add in val data as test data, must split first

@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    itrain: pd.DataFrame
    itrainlab: pd.DataFrame
    itest: pd.DataFrame
    itestlab: pd.DataFrame

    #Training set
    train: np.ndarray = field(init=False)
    train_labels: np.ndarray = field(init=False)

    #Validation set
    val:np.ndarray = field(init=False)
    val_labels: np.ndarray = field(init=False)

    #Test Set
    test: np.ndarray = field(init=False)
    test_labels: np.ndarray = field(init=False)

    def __post_init__(self,rng):
        self.train = self.itrain.iloc[:50000].values.reshape(-1,28,28,1)
        self.train_labels = self.itrainlab.iloc[:50000].to_numpy()

        self.val = self.itrain.iloc[50000:].values.reshape(-1,28,28,1)
        self.val_labels = self.itrainlab.iloc[50000:].to_numpy()

        self.test = self.itest.values.reshape(-1,28,28,1)
        self.test_labels = self.itestlab.to_numpy()


    def get_batch(self, rng, batch_size):
        choices = rng.choice(self.index, size = batch_size)
        return self.x[choices], self.y[choices], self.tclass[choices]

    def get_spirals(self):
        return self.x1, self.y1, self.x2, self.y2

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 50000, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 50, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 5000, "Number of forward/backward pass iterations")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate/initial step size")
flags.DEFINE_integer("random_seed", 31415, "Random seed for reproducible results")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")


#feed in x, y, and category as training data, predict boundaries with multilayer perceptron
def Model():
        #variables to be tuned in inits, the layers that will be made by layer class when we call model in main
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (32,32,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (32,32,3)))
    model.add(layers.MaxPooling2D(2,2))

    #add Dense layers for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # size 10 output layer for 10 classes
    model.add(layers.Dense(10, activity_regularizer = regularizers.L2(0.01)))

     #model has optimzeer and loss function built in now
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

def main():

    #parse flags before we use them
    FLAGS(sys.argv)

    #set seed for reproducible results
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2) #spawn 2 sequences for 2 threads
    np_rng =np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    #make data into pandas df
    images_df = pd.read_csv('./images.csv')
    labels_df = pd.read_csv('./labels.csv')

    print(images_df.shape, labels_df.shape)
    test_images = pd.read_csv('./testimages.csv')
    test_labels = pd.read_csv('./testlabels.csv')

    #call Data class to properly shape data
    data = Data(rng = np_rng, itrain = images_df, itrainlab = labels_df, itest = test_images, itestlab = test_labels)

    print(data.train.shape, data.train_labels.shape)
    model = Model()
    history = model.fit(data.train, data.train_labels, epochs=10,
        validation_data=(data.val, data.val_labels))



#PLOTTING

if __name__ == "__main__":
    main()
