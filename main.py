import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

# classify MNIST digits with convolutional neural network
# objective 95.5% accuracy

#split train data into train and val

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

#add in val data as test data, must split first
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
