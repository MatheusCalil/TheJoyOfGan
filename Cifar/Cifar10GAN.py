#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thanks to 
https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

https://www.tensorflow.org/tutorials/generative/dcgan

@author: matheus.faleiros
"""

from PIL import Image
import os, numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 16

INPUT_SHAPE = [32,32,3]

def load_dataset(folder="/Users/matheus.faleiros/Documents/bobross"):
    (X_train, y_train), (X_test, _) = tf.keras.datasets.cifar10.load_data()  
    # Select Cars
    X_train = X_train[y_train[:,0]==1]
    return X_train


def display_sample(images, index):
    Image.fromarray(images[index], 'RGB').show()

def real_batch(images):
    train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(123).batch(BATCH_SIZE)
    return train_dataset

def preprocessing(images):
    #Preprocessing will only normalize the images from [0,255] to [-1,1]
    images = (np.array(images, dtype='uint8') - 127.5)/127.5
    
    return images

############# Modeling #############

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                     input_shape=INPUT_SHAPE))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(248, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')) #NEW
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(4*4*256,activation="relu",input_shape=(100,)))
    model.add(layers.Reshape((4,4,256)))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256,kernel_size=3,padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256,kernel_size=3,padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
   
    # Output resolution, additional upsampling
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128,kernel_size=3,padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))

    # Final CNN layer
    model.add(layers.Conv2D(3,kernel_size=3,padding="same"))
    model.add(layers.Activation("tanh"))


    return model


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images,generator,discriminator):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
      
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss


def train_model(batch_images, epoch,generator,discriminator):
    fixed_seed = tf.random.normal([1, 100])
    for epoch in tqdm(range(epoch)):
            gen_loss_list = []
            disc_loss_list = []
            
            for image_batch in batch_images:
                gen_loss, disc_loss = train_step(image_batch,generator,discriminator)
                gen_loss_list.append(gen_loss)
                disc_loss_list.append(disc_loss)
                
            g_loss = sum(gen_loss_list) / len(gen_loss_list)
            d_loss = sum(disc_loss_list) / len(disc_loss_list)
            print(f"""Epoch {epoch+1} gen_loss = {g_loss} disc_loss = {d_loss}""")
            #Generate one image per epoch
            if epoch%10 == 0:
                generated_image = generator(fixed_seed, training=False)
                display_sample(np.array(generated_image[0, :, :, :]* 127.5 + 127.5, dtype='uint8').reshape((1,32,32,3)),0)

def discriminator_auc(discriminator, batch_to_auc, ground_truth):
    m = tf.keras.metrics.AUC()
    predictions = [discriminator(item.reshape((1,32,32,3)), training=False)[0] for item in batch_to_auc]
    #print(predictions)
    m.update_state(ground_truth,predictions)
    return m.result().numpy()
    
#test pipeline
images = load_dataset()

display_sample(np.array(images, dtype='uint8'), 400)

processed_images = preprocessing(images)

batch_images = real_batch(processed_images)

display_sample(np.array(processed_images, dtype='uint8'),0)

display_sample(np.array(processed_images*127.5+127.5, dtype='uint8'),0)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator.summary()

discriminator.summary()

train_model(batch_images, 200,generator,discriminator)

#getting one batch of positive and negative to evaluation with AUC
noise = tf.random.normal([BATCH_SIZE, 100])
generated_image = generator(noise, training=False)
ground_truth = []
batch_to_auc = []
for item in list(batch_images.as_numpy_iterator())[0]:
    batch_to_auc.append(item)
    ground_truth.append(1)
for item in list(generated_image.numpy()):
    batch_to_auc.append(item)
    ground_truth.append(0)
    

print(discriminator_auc(discriminator, batch_to_auc, ground_truth))
    
display_sample(np.array(generated_image[0, :, :, :]* 127.5 + 127.5, dtype='uint8').reshape((1,32,32,3)),0)


for i in range(10):
    image = np.array(generated_image[i, :, :, :]* 127.5 + 127.5, dtype='uint8').reshape((1,32,32,3))
    #Image.fromarray(image[0], 'RGB').show()
    Image.fromarray(image[0], 'RGB').save(f"/Users/matheus.faleiros/Documents/TheJoyOfGan/{i}.png")
    
for i in range(10):
    image = np.array(images, dtype='uint8')
    #Image.fromarray(image[i*i], 'RGB').show()
    Image.fromarray(image[i*i], 'RGB').save(f"/Users/matheus.faleiros/Documents/TheJoyOfGan/{i+10}.png")


