#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from absl import flags
from tqdm import trange

from dataclasses import dataclass, field, InitVar
from tensorflow.keras import layers, models

# classify MNIST digits with convolutional neural network
# objective 95.5% accuracy

#split train data into train and val

#add in val data as test data, must split first
history = model.fit(train_images, train_labels, epochs=10,
        validation_data=(test_images, test_labels))

@dataclass
class Data:
    rng: InitVar[np.random.Generator]
    num_samples: int

    x1: np.ndarray = field(init=False)
    y1: np.ndarray = field(init=False)
    x2: np.ndarray = field(init=False)
    y2: np.ndarray = field(init=False)

    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    tclass: np.ndarray = field(init=False)

    def __post_init__(self, rng):
        #return evenly spaced values from 0 to num_samples
        self.index = np.arange(self.num_samples)

        #Generate spiral data, vary r while moving through x and y
        half_data = self.num_samples // 2
        clean_r = np.linspace(1,12, half_data)
        clean_theta1=np.linspace(6,2.5, half_data)
        clean_theta2 = np.linspace(5, 1.5, half_data)

        #make noisy draw from normal dist
        r = rng.normal(loc= clean_r, scale =0.1)
        theta1 = rng.normal(loc= clean_theta1, scale =0.1)
        theta2 = rng.normal(loc= clean_theta2, scale =0.1)

        #had to astype everything to float 32 due to casting error later
        self.x1=r*np.cos(np.pi*theta1).astype(np.float32)
        self.y1=r*np.sin(np.pi*theta1).astype(np.float32)

        self.x2=r*np.cos(np.pi*theta2).astype(np.float32)
        self.y2=r*np.sin(np.pi*theta2).astype(np.float32)

        #make output data
        class0 = np.zeros(half_data, dtype=np.float32)
        class1 = np.ones(half_data, dtype=np.float32)

        #Combine data vectors to make whole training and output set
        self.x = np.append(self.x1,self.x2).astype(np.float32)
        self.y = np.append(self.y1,self.y2).astype(np.float32)
        self.tclass = np.append(class0,class1).astype(np.float32)

    def get_batch(self, rng, batch_size):
        choices = rng.choice(self.index, size = batch_size)
        return self.x[choices], self.y[choices], self.tclass[choices]

    def get_spirals(self):
        return self.x1, self.y1, self.x2, self.y2

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 500, "Number of samples in dataset")
flags.DEFINE_integer("batch_size", 50, "Number of samples in batch")
flags.DEFINE_integer("num_iters", 5000, "Number of forward/backward pass iterations")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate/initial step size")
flags.DEFINE_integer("random_seed", 31415, "Random seed for reproducible results")
flags.DEFINE_float("sigma_noise", 0.5, "Standard deviation of noise random variable")

#i attempted to matmul my layers which i made by passing input through activation function and then matmuling, but i kept getting lost in the sauce so i turned to the internet

#make layer class
#https://www.tensorflow.org/guide/core/mlp_core#multilayer_perceptron_mlp_overview
#document sent to me by Husam

#must initialize weights properly to prevent activation outputs from becoming too large or small, use xavier init method to do so
def xavier_init(shape):
        in_dim, out_dim = shape
        #in_dim= tf.cast(in_dim, tf.int32)
        #out_dim = tf.cast(out_dim, tf.int32)
        xavier_lim = tf.sqrt(6.)/tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
        weight_vals =tf.random.uniform(shape=(in_dim, out_dim), minval=-xavier_lim, maxval=xavier_lim, seed =22)
        return weight_vals

class Layer(tf.Module):

    def __init__(self, out_dim, weight_init = xavier_init, activation = tf.nn.relu):
        self.out_dim=out_dim
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self,x):
        if not self.built:
            #get input dimension by first input layer
            self.in_dim = x.shape[1]
            #get weight and bias using xavier scheme
            self.w = tf.Variable(xavier_init(shape=(self.in_dim, self.out_dim)))
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
            self.built = True
            print(x)
            print(self.w)

        #float64w = tf.cast(self.w, tf.float64)
        #float64b = tf.cast(self.b, tf.float64)
        z = x @ self.w + self.b
        return self.activation(z)


#feed in x, y, and category as training data, predict boundaries with multilayer perceptron
class Model(tf.Module):
        #variables to be tuned in inits, the layers that will be made by layer class when we call model in main
    def __init__(self, layers):
        self.layers = layers

    @tf.function
    def __call__(self, x, preds = False):
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
     model.add(layers.Dense(10))

     model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


def main():

    #parse flags before we use them
    FLAGS(sys.argv)

    #set seed for reproducible results
    seed_sequence = np.random.SeedSequence(FLAGS.random_seed)
    np_seed, tf_seed = seed_sequence.spawn(2) #spawn 2 sequences for 2 threads
    np_rng =np.random.default_rng(np_seed)
    tf_rng = tf.random.Generator.from_seed(tf_seed.entropy)

    #Generate data

    data = Data(np_rng, FLAGS.num_samples)

    #call model and feed in x, y, and true category as training data, predict boundaries with multilayer perceptron
    model = Model([
        Layer(200),
        Layer(100),
        Layer(1, activation=tf.math.sigmoid)])

    optimizer = tf.optimizers.Adam(learning_rate = FLAGS.learning_rate)

    #makes the sexy bar that shows progress of our training
    bar = trange(FLAGS.num_iters)

    for i in bar:
        with tf.GradientTape() as tape:
            x,y, tclass = data.get_batch(np_rng, FLAGS.batch_size)
            #make coordinates into tuple so xavier init can get first input dim, make batch_size num of columns
            xycoord = np.append(tf.squeeze(x), tf.squeeze(y)).reshape(2,FLAGS.batch_size).T
            tclass = tf.squeeze(tclass)
            class_hat = tf.squeeze(model(xycoord, tf_rng))
            #add tiny constant so loss is not 0
            temp_loss = tf.reduce_mean((-tclass*tf.math.log(class_hat)+1e-25) -((1-tclass)*tf.math.log(1-class_hat) +1e-25))
            l2_reg_const = 0.001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in model.trainable_variables ])
            loss = temp_loss + l2_reg_const
            print(loss)



        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        #bar.set_description(f"Loss @ {i} => {loss.numpy():0.3f}")
        bar.refresh()

    #due to the time constraint and my inability to make my plot work, this plot is heavily baseed on Husam's plotting section
    #PLOTTING
    N_points = 100
    axis = np.linspace(-15,15, 1000).astype(np.float32)
    x_ax, y_ax = np.meshgrid(axis,axis)

    coords = np.vstack([x_ax.ravel(), y_ax.ravel()]).T
    y = model(coords, tf_rng)
    output = tf.squeeze(y)

    plt.figure()
    num_samples = FLAGS.num_samples
    plt.plot(data.x[:num_samples//2], data.y[:num_samples//2], "o", color="red")
    plt.plot(data.x[num_samples//2:], data.y[num_samples//2:], "o", color="blue")
    plt.legend(["Data 0", "Data 1"])
    #make boundary point at 0.5 bc aligns with sigmoid output layer
    plt.contourf(x_ax, y_ax, output.numpy().reshape(1000, 1000), [0, 0.5, 1], colors=["lightcoral", "steelblue"])
    plt.title("Spiral Data")

    plt.tight_layout()
    plt.savefig("./spiral.pdf")


    #x1,y1,x2,y2 = data.get_spirals()
    #fig, ax = plt.subplots(1,1, figsize=(15,15), dpi = 200)
    #ax.set_title("Spirals")
    #ax.set_xlabel("Spiral Radius")
    #ax.set_ylabel("Spiral Radius")
    #ax.set_xlim(-15, 15)
    #ax.set_ylim(-15, 15)
    #ax.plot(x1, y1, "o", x2, y2, "o")

   # plt.savefig("./spirals1test.pdf

if __name__ == "__main__":
    main()
