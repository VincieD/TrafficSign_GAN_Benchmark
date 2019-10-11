from __future__ import print_function, division

import os
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from functools import partial
import keras.backend as K
from keras.layers import Layer, InputSpec

import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import numpy as np
from scipy import misc
import time

from dataset_signs import DATA_SET
from calcualte_fid import calculate_FID

#from keras.utils import plot_model
#from keras_sequential_ascii import keras2ascii


def swish(x):
    beta = 1.0 #1, 1.5 or 2
    return beta * x * K.sigmoid(x)


class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)

        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        global batch_size
        alpha = K.random_uniform((batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self, numOfClasses, batch_size=32, grayScale=True, leaky=False, swish=False, attention_middle=False, attention_last=False, conditional=False):
        self.img_rows = 64
        self.img_cols = 64
        if grayScale == True:
            self.channels = 1
        else:
            self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        #self.latent_dim = 100
        self.num_classes = numOfClasses
        self.latent_dim = 80
        self.losslog = []
        self.batch_size = batch_size
        self.grayScale = grayScale
        self.leaky = leaky
        self.swish = swish
        self.attention_middle = attention_middle
        self.attention_last = attention_last
        self.conditional = conditional

        self.imagPath = "images"

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 2  #original 2

        if self.leaky:
            optimizer_gen = Adam(lr=5e-05, beta_1=0.0, beta_2=0.7, epsilon=1e-08) # working
            optimizer_cri = Adam(lr=2e-04, beta_1=0.0, beta_2=0.7, epsilon=1e-08) # working
        elif self.swish:
            optimizer_gen = Adam(lr=5e-05, beta_1=0.0, beta_2=0.7, epsilon=1e-08) # working
            optimizer_cri = Adam(lr=2e-04, beta_1=0.0, beta_2=0.7, epsilon=1e-08) # working
        else:
            optimizer_gen = Adam(lr=5e-05, beta_1=0.0, beta_2=0.7, epsilon=1e-08) # working
            optimizer_cri = Adam(lr=2e-04, beta_1=0.0, beta_2=0.7, epsilon=1e-08) # working


        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #keras2ascii(self.generator)
        #plot_model(self.generator, to_file='generator_model.png')

        #keras2ascii(self.critic)
        #plot_model(self.critic, to_file='generator_model.png')

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # ADDING LABEL TO MAKE IT DWGAN
        if self.conditional:
            label = Input(shape=(1,))
            # Generate image based of noise (fake sample)
            fake_img = self.generator([z_disc, label])
            # Discriminator determines validity of the real and fake images
            fake = self.critic([fake_img, label])

            valid = self.critic([real_img, label])
            # Construct weighted average between real and fake images
            interpolated_img = RandomWeightedAverage()([real_img, fake_img])
            # Determine validity of weighted sample
            validity_interpolated = self.critic([interpolated_img, label])

            # Use Python partial to provide loss function with additional
            # 'averaged_samples' argument
            partial_gp_loss = partial(self.gradient_penalty_loss,
                              averaged_samples=interpolated_img)
            partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

            self.critic_model = Model(inputs=[real_img, label, z_disc], outputs=[valid, fake, validity_interpolated])
            self.critic_model.compile(loss=[self.wasserstein_loss,
                                                  self.wasserstein_loss,
                                                  partial_gp_loss],
                                            optimizer=optimizer_cri,
                                            loss_weights=[1, 1, 10])
        else:
            fake_img = self.generator(z_disc)
            # Discriminator determines validity of the real and fake images
            fake = self.critic(fake_img)
            valid = self.critic(real_img)


            # Construct weighted average between real and fake images
            interpolated_img = RandomWeightedAverage()([real_img, fake_img])
            # Determine validity of weighted sample
            validity_interpolated = self.critic(interpolated_img)

            # Use Python partial to provide loss function with additional
            # 'averaged_samples' argument
            partial_gp_loss = partial(self.gradient_penalty_loss,
                              averaged_samples=interpolated_img)
            partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

            self.critic_model = Model(inputs=[real_img, z_disc],
                                outputs=[valid, fake, validity_interpolated])

            self.critic_model.compile(loss=[self.wasserstein_loss,
                                                  self.wasserstein_loss,
                                                  partial_gp_loss],
                                            optimizer=optimizer_cri,
                                            loss_weights=[1, 1, 10])

        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))

        if self.conditional:
            # add label to the input
            label = Input(shape=(1,))
            # Generate images based of noise
            #img = self.generator(z_gen)
            img = self.generator([z_gen, label])
            # Discriminator determines validity
            valid = self.critic([img, label])
            # Defines generator model
            self.generator_model = Model([z_gen, label], valid)
        else:
            # Generate images based of noise
            img = self.generator(z_gen)
            # Discriminator determines validity
            valid = self.critic(img)
            # Defines generator model
            self.generator_model = Model(z_gen, valid)


        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer_gen)
        

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def init_weights(self, shape, name=None):
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


    def init_bias(self, shape, name=None):
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        if self.leaky == True:
            model.add(Dense(256 * 4 * 4, input_dim=self.latent_dim, kernel_initializer=self.init_weights))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Dense(256 * 6 * 6, activation = swish, input_dim=self.latent_dim, kernel_initializer='he_normal'))
            model.add(Dense(256 * 4 * 4, activation = swish, input_dim=self.latent_dim, kernel_initializer=self.init_weights))
        else:
            model.add(Dense(256 * 4 * 4, input_dim=self.latent_dim, kernel_initializer=self.init_weights))
            model.add(Activation("relu"))

        model.add(Reshape((4, 4, 256)))

        if self.leaky == True:
            model.add(Conv2D(256, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(256, kernel_size=4, activation=swish, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(256, kernel_size=4, activation=swish, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Conv2D(256, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        model.add(UpSampling2D())

        if self.leaky == True:
            model.add(Conv2D(128, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(128, kernel_size=4, activation=swish, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(128, kernel_size=4, activation=swish, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))      
        else:
            model.add(Conv2D(128, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))


        model.add(UpSampling2D())

        if self.leaky == True:
            model.add(Conv2D(64, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(64, kernel_size=4, activation=swish, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(64, kernel_size=4, activation=swish, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Conv2D(64, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        # Adding Attendtion layer in the middle
        if self.attention_middle:
            model.add(Attention(64))
        model.add(UpSampling2D())

        if self.leaky == True:
            model.add(Conv2D(32, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(32, kernel_size=4, activation=swish, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(32, kernel_size=4, activation=swish, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Conv2D(32, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        if self.attention_last:
            model.add(Attention(32))
        model.add(UpSampling2D())

        if self.leaky == True:
            model.add(Conv2D(16, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(32, kernel_size=4, activation=swish, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(16, kernel_size=4, activation=swish, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Conv2D(16, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=4, padding="same", kernel_initializer=self.init_weights))
        model.add(Activation("tanh"))

        model.summary()

        #keras2ascii(model)
        #plot_model(model, to_file='generator.png', show_shapes=True, show_layer_names=True)

        noise = Input(shape=(self.latent_dim,))
        if self.conditional:
            # Adding fro CWGAN
            label = Input(shape=(1,), dtype='int32')
            label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

            model_input = multiply([noise, label_embedding])
            img = model(model_input)

            return Model([noise, label], img)
        else:
            img = model(noise)

            return Model(noise, img)


    def build_critic(self):

        model = Sequential()

        if self.conditional:
            if self.leaky == True:
                model.add(Dense(np.prod(self.img_shape), input_dim=np.prod(self.img_shape), kernel_initializer=self.init_weights))
                model.add(LeakyReLU(alpha=0.2))
            elif self.swish == True:
                #model.add(Dense(np.prod(self.img_shape), activation = swish, input_dim=np.prod(self.img_shape), kernel_initializer='he_normal'))
                model.add(Dense(np.prod(self.img_shape), activation = swish, input_dim=np.prod(self.img_shape), kernel_initializer=self.init_weights))
            else:
                model.add(Dense(np.prod(self.img_shape), input_dim=np.prod(self.img_shape), kernel_initializer=self.init_weights))
                model.add(Activation("relu"))

            model.add(Reshape((64, 64, 3)))

        if self.conditional:
            if self.leaky == True:
                model.add(Conv2D(16, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
                model.add(LeakyReLU(alpha=0.2))
            elif self.swish == True:
                #model.add(Conv2D(16, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer='he_normal'))
                model.add(Conv2D(16, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer=self.init_weights))
            else:
                model.add(Conv2D(16, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
                model.add(Activation("relu"))
        else:
            if self.leaky == True:
                model.add(Conv2D(16, kernel_size=3, input_shape=self.img_shape, strides=1, padding="same", kernel_initializer=self.init_weights))
                model.add(LeakyReLU(alpha=0.2))
            elif self.swish == True:
                #model.add(Conv2D(16, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer='he_normal'))
                model.add(Conv2D(16, kernel_size=3, input_shape=self.img_shape, activation=swish, strides=1, padding="same", kernel_initializer=self.init_weights))
            else:
                print(self.img_shape)
                model.add(Conv2D(16, kernel_size=3, input_shape=self.img_shape, strides=1, padding="same", kernel_initializer=self.init_weights))
                model.add(Activation("relu"))

        model.add(Dropout(0.25))
        # Adding Attendtion layer in the middle
        if self.attention_last:
            model.add(Attention(16))

        if self.leaky == True:
            model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            #model.add(Conv2D(32, kernel_size=3, input_shape=self.img_shape, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(32, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(32, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer=self.init_weights))
            #model.add(Conv2D(32, kernel_size=3, input_shape=self.img_shape, activation = swish, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            #model.add(Conv2D(32, kernel_size=3, input_shape=self.img_shape, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        model.add(Dropout(0.25))
        # Adding Attendtion layer in the middle
        if self.attention_middle:
            model.add(Attention(32))

        if self.leaky == True:
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(32, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(64, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        model.add(Dropout(0.25))


        if self.leaky == True:
            model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(64, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(128, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer=self.init_weights))
        else:
            model.add(Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(Activation("relu"))

        model.add(Dropout(0.25))


        if self.leaky == True:
            model.add(Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        elif self.swish == True:
            #model.add(Conv2D(128, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer='he_normal'))
            model.add(Conv2D(256, kernel_size=3, activation = swish, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_initializer=self.init_weights))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(576))
        model.add(Dense(1))

        model.summary()

        #keras2ascii(model)
        #plot_model(model, to_file='generator.png', show_shapes=True, show_layer_names=True) 

        img = Input(shape=self.img_shape)

        if self.conditional:
            # Prepared for conditional Critic
            label = Input(shape=(1,), dtype='int32')

            label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape)) (label))

            flat_img = Flatten()(img)
            model_input = multiply([flat_img, label_embedding])

            validity = model(model_input)
            return Model([img,label], validity)
        else:
            validity = model(img)

            return Model(img, validity)


    def train(self, numPos, epochs, sample_interval=100):

        X_train = dataset.X_train_pos[:,:,:,:]
        d_loss = []
        g_loss = []
        index = 0

        if self.conditional:
            Y_train = dataset.Y_train[:] - 1

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty
        start_time = time.time()
        for epoch in range(epochs):

            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, numPos, self.batch_size)
                if self.conditional:
                    imgs, labels = X_train[idx], Y_train[idx]
                else:
                    imgs = X_train[idx]

                # Sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # Train the critic
                if self.conditional:
                    if index % self.n_critic == 0:
                        d_loss.append(self.critic_model.train_on_batch([imgs, labels, noise], [valid, fake, dummy])[0])
                        index=0
                else:
                    if index % self.n_critic == 0:
                        d_loss.append(self.critic_model.train_on_batch([imgs, noise], [valid, fake, dummy])[0])
                        index=0
                index = index + 1
            # ---------------------
            #  Train Generator
            # ---------------------
            if self.conditional:
                # Condition on labels
                sampled_labels = np.random.randint(0, self.num_classes, self.batch_size).reshape(-1, 1)

                g_loss.append(self.generator_model.train_on_batch([noise, sampled_labels], valid))
            else:
                g_loss.append(self.generator_model.train_on_batch(noise, valid))

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[-1], g_loss[-1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                plt.plot(d_loss)
                plt.plot(g_loss)
                plt.legend(['discriminator, lr=2e-04, beta_2=0.5','generator_loss, lr=6e-04, beta_2=0.5'])
                plt.savefig(self.imagPath +'/loss.png')
                plt.clf()
                print('Time ellapsed: %f s' % float(time.time() - start_time))
                delta = float(time.time() - start_time)
                remainingMinutes =  (epochs - epoch) / 100 * delta / 60
                print('Remaining minutes to end: %f min' % remainingMinutes)
                start_time = time.time()


    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        #noise = truncated_noise_sample(batch_size=r * c, dim_z=self.latent_dim, truncation=truncation)
        if self.conditional:
            sampled_labels = np.random.randint(0, self.num_classes,r * c).reshape(-1, 1)
            gen_imgs = self.generator.predict([noise,sampled_labels])
        else:
            gen_imgs = self.generator.predict(noise)

        if not os.path.exists(self.imagPath):
            os.mkdir(self.imagPath)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                #axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                if self.conditional:
                    axs[i,j].set_title("Sample: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.imagPath + "/genSigns_col_%d.png" % epoch)
        plt.close()

        # generate images for FID ind IS purpose
        FID = True
        if (FID == True) and ((epoch == 50000) or (epoch == 100000)):
            how_many = 50
            dirName = os.path.join(self.imagPath, str(epoch))
            print ("Generating images for FID in epoch: {}".format(epoch))
            if not os.path.exists(dirName):
                os.mkdir(dirName)

            for index in range(how_many):
                noise = np.random.normal(0, 1, (1, self.latent_dim))
                
                if self.conditional:
                    sampled_labels = np.random.randint(0, self.num_classes, how_many).reshape(-1, 1)
                    figure = self.generator.predict([noise,sampled_labels])
                else:
                    figure = self.generator.predict(noise)
                figure = 0.5 * figure + 1
                #figure = normalize(figure, axis=1)
                out = os.path.join(dirName, str(index) + ".png")
                misc.imsave(out, figure[0,:,:,:])

             # ---------------------
            # Saving model to JSON
            # serialize model to JSON
            path_g = os.path.join(dirName, "_gen_model_LeakyRelu_selfAtt_middle")
            name_g_model = os.path.join(path_g, ".json") 
            name_g_weights = os.path.join(path_g, ".h5") 

            model_json = self.generator.to_json()
            with open(name_g_model, "w") as json_file:
                 json_file.write(model_json)
            # # serialize weights to HDF5
            self.generator.save_weights(name_g_weights)
            print("Generator model saved to disk")


if __name__ == '__main__':

    realImage_Path = "dataSet_mini"

    dataset = DATA_SET("/storage/plzen1/home/vincie/wgan/dataSet_mini",
        grayScale=False, 
        labels=False,
        img_rows= 64,
        img_cols = 64)

    batch_size = 64

    numPos, num_classes = dataset.loadImages()

    wganRelU = WGANGP(num_classes, batch_size=batch_size, grayScale=False, leaky=False, swish=False, attention_middle=False, attention_last=False, conditional=False)
    wganRelU.imagPath = "images/class1_64x64_batch128_cr4_adam_layers5_RelU"
    wganRelU.train(numPos, epochs=10000, sample_interval=100)

    wganSwish = WGANGP(num_classes, batch_size=batch_size, grayScale=False, leaky=False, swish=True, attention_middle=False, attention_last=False, conditional=False)
    wganSwish.imagPath = "images/class1_64x64_batch128_cr4_adam_layers5_Swish"
    wganSwish.train(numPos, epochs=10000, sample_interval=100)

    wganLeakyRelUSelfAt_middle = WGANGP(num_classes, batch_size=batch_size, grayScale=False, leaky=True, swish=False, attention_middle=True, attention_last=False, conditional=False)
    wganLeakyRelUSelfAt_middle.imagPath = "images/class1_64x64_batch128_cr4_adam_layers5_LeakyRelu_selfAtt_middle"
    wganLeakyRelUSelfAt_middle.train(numPos, epochs=10000, sample_interval=100)

    wganLeakyRelUSelfAt_last = WGANGP(num_classes, batch_size=batch_size, grayScale=False, leaky=True, swish=False, attention_middle=True, attention_last=False, conditional=False)
    wganLeakyRelUSelfAt_last.imagPath = "images/class1_64x64_batch128_cr4_adam_layers5_LeakyRelu_selfAtt_last"
    wganLeakyRelUSelfAt_last.train(numPos, epochs=10000, sample_interval=100)
    
    generatedImages_Path = []
    generatedImages_Path.append(wganRelU.imagPath, wganSwish.imagPath, wganLeakyRelUSelfAt_middle.imagPath, wganLeakyRelUSelfAt_last.imagPath)

    calculate_FID(realImage_Path, generatedImages_Path)
