#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Principais referencias para esse codigo

https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

https://www.tensorflow.org/tutorials/generative/dcgan

https://arxiv.org/pdf/1903.06259.pdf

@author: matheus.faleiros
"""

from PIL import Image
import os, numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 15 #Tamanho do Batch

INPUT_SHAPE = [128,128,3] #Shape das imagens

def load_dataset(folder=""):
    """
        Função que carrega o dataset em formato RGB com altura e largura 128x128 (3,128,128)
        
        Input:
            folder [String] - Caminho das imagens. Elas devem começar com painting.
        
        Output:
            ims [List] - lista contendo as imagens do diretorio informado
    """
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB").resize((128,128))) #resize to get 32,32,3 and keep things easy

    ims = [read(os.path.join(folder, filename))  for filename in os.listdir(folder) if "painting" in filename]  
    
    return ims


def display_sample(images, index):
    """
     Funçao para dar display em uma imagem e exibir ela na tela
     
     Inputs:
         images [Lista] - contem de N imagens, por exemplo, output de load_dataset
         index [int] -  indice da imagem na lista para ser exibida
         
    Output:
        None
    """
    Image.fromarray(images[index], 'RGB').show()

def real_batch(images):
    """
        Cria o batch de imagens reais dado as imagens lidas do diretório
        
        Input:
            images [list] - Imagens lidas do diretorio de origem ou processadas pela função preprocessing
        Output
            train_dataset [Tensor] - Tensor contendo batchs do tamanho da global BATCH_SIZE
    """
    train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(123).batch(BATCH_SIZE)
    return train_dataset

def preprocessing(images):
    #Preprocessing will only normalize the images from [0,255] to [-1,1]
    images = (np.array(images, dtype='uint8') - 127.5)/127.5
    
    return images

############# Modeling #############

def make_discriminator_model():
    """
        Função que gerar a arquitetura do modelo discriminador.
        SN GAN: cada bloco contem 2 convolucoes, uma 3x3 stride 1 e outra 4x4 de stride 2
        Os numero de mapas vai de 64 para 128 para 256 para 512.
        Dropout adicionados parar evitar overfitting e ajudar no treino.
        
        Input:
            
        Output:
            model [keras.Sequential] - Modelo sequencial de CNN
    """

    model = tf.keras.Sequential()
    
    #Bloco 1
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',input_shape=INPUT_SHAPE))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    #Bloco 2
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    #Bloco 3
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    
    #Final Convolution
    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='sigmoid')) #adicionado
    model.add(layers.Dense(1, activation='linear')) #trocado sigmoid por linear

    return model

def make_generator_model():
    """
        Função que cria o gerador
        SN-GAN
        
        Input:
            
        Output:
            model [keras.Sequential()] - Modelo CNN de descovolução
    """
    #SN-GAN Generator
    #MODIFICAÇÕES: Adicionado uma conv a  mais
    model = tf.keras.Sequential()

    model.add(layers.Dense(16*16*512,activation="relu",input_shape=(100,)))
    model.add(layers.Reshape((16,16,512)))

    #Stride 2 4x4 convolution
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256,kernel_size=4,padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    #model.add(layers.Dropout(0.25))

    
    #Stride 2 4x4 convolution
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128,kernel_size=4,padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    #model.add(layers.Dropout(0.25))

    #Stride 2 4x4 convolution
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64,kernel_size=4,padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    #model.add(layers.Dropout(0.25))

    #Stride 1 3x3 covolution
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

#Cria os otimizadores para os dois modelos.
generator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images,generator,discriminator):
    """
        Passo de treinamento do generator e discriminator.
        A anotation @tf.function faz com que a função seja pre-compilada, aumentando a eficiência.
        
        Input:
            images [Tensor] - Um unico batch de imagens geradas em real_batch
            generator [keras.Sequential] - Modelo gerador
            discriminator [keras.Sequential] - Modelo discriminador
        Output
            gen_loss [float] - float contendo a loss do gerador nesse passo de treino
            disc_loss [float] - float contendo a loss do discriminador nesse passo de treino
    """
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
    """
        Processo de treinamento total.
        Fixamos uma seed para avaliar o generator visualmente a cada 10 epochs
        Calculamos a loss média do batch para os dois modelos, para acompanhamento.
        Os objetos dos modelos são alterados por referência, portanto não precisam ser retornados na função.
        Input:
            batch_mages [Tensor] - Batch de imagens geradas em real_batch
            epoch [int] - numero de epochs a serem treinadas
            generator [keras.Sequential] - Modelo gerador
            discriminator [keras.Sequential] - Modelo discriminador
        Output:
            None
    """
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
                display_sample(np.array(generated_image[0, :, :, :]* 127.5 + 127.5, dtype='uint8').reshape((1,128,128,3)),0)

def discriminator_auc(discriminator, batch_to_auc, ground_truth):
    #Calcula a AUC do discriminador. Não foi utilizada.
    m = tf.keras.metrics.AUC()
    predictions = [discriminator(item.reshape((1,128,128,3)), training=False)[0] for item in batch_to_auc]
    #print(predictions)
    m.update_state(ground_truth,predictions)
    return m.result().numpy()
    
    
#test pipeline
images = load_dataset()

#display_sample(np.array(images, dtype='uint8'), 0)

processed_images = preprocessing(images)

batch_images = real_batch(processed_images)

#display_sample(np.array(processed_images, dtype='uint8'),0)

#display_sample(np.array(processed_images*127.5+127.5, dtype='uint8'),0)

generator = make_generator_model()
discriminator = make_discriminator_model()

#Display a arquitetura
generator.summary()
discriminator.summary()

#Treino
train_model(batch_images, 200,generator,discriminator)

#Geramos um batch de imagens artificiais com o generator treinado
noise = tf.random.normal([BATCH_SIZE, 100])
generated_image = generator(noise, training=False)
    
#Vamos agora salvar as imagens
for i in range(10):
    image = np.array(generated_image[i, :, :, :]* 127.5 + 127.5, dtype='uint8').reshape((1,128,128,3))
    Image.fromarray(image[0], 'RGB').save(f"{i}.png")
    
for i in range(10):
    image = np.array(images, dtype='uint8')
    Image.fromarray(image[i*i], 'RGB').save(f"{i+10}.png")
    


