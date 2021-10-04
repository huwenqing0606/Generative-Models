"""
The GAN training encapsulated in a class
Environment: Keras and Tensorflow 1.14
Reference: Goodfellow, I. et al, Generative Adversarial Nets, NIPS 2014
Author: Wenqing Hu (Missouri S&T)
"""

from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np

class GAN:
    # constructor function
    def __init__(self,
                 generator, # the generator network
                 discriminator, # the discriminator network
                 noise_dim, # the input noise dimension, i.e. the hidden variable Z
                 noise_type, # the type of the input noise, currently can choose 'uniform' or 'gaussian'
                 ):
        self.G = generator
        self.D = discriminator
        self.noise_dim = noise_dim
        self.noise_type = noise_type
    

    # The generator network composed with the discriminator network
    # the D(G(z)) in Goodfellow et al NIPS 2014
    def generator_containing_discriminator(self, generator, discriminator):
        # choose a keras sequential model
        model = Sequential()
        # first go through the generator network
        model.add(generator)
        # set the discriminator to be not trainable
        discriminator.trainable = False
        # add the discriminator network
        model.add(discriminator)
        # build D(G(z))
        return model


    # The noise generator, given noise dimension and noise type
    # num: the number of i.i.d. noise samples
    def noise_generator(self, num):
        if self.noise_type == 'uniform':
            noise = np.random.uniform(-1, 1, size=(num, self.noise_dim))
        elif self.noise_type == 'gaussian':
            noise = np.random.uniform(-1, 1, size=(num, self.noise_dim))
        else:
            print('Class GAN: Noise type error! Input uniform or gaussian \n')
            noise = None
        return noise


    # train via the GAN algorithm
    # Input Variables:
    #   train_inputs: the training input data, should be an array where each element is an input array, i.e. array([input, input,...])
    #   G_optimizer : the optimizer for generator network
    #   D_optimizer : the optimizer for discriminator 
    #   BATCH_SIZE  : the sampling batch size
    #   sampling    : the method of sampling batches, choose 'deterministic_sweep' or 'random_uniform'
    #   num_epoch   : the number of epochs in training
    #   num_iter    : the number of iterations in each epoch
    def train(self, train_inputs, G_optimizer, D_optimizer, BATCH_SIZE, sampling, num_epoch, num_iter):
        # build G(D(z))
        G = self.G
        D = self.D
        D_compose_G = self.generator_containing_discriminator(G, D)
        # compile the networks using binary cross-entropy loss
        G.compile(loss='binary_crossentropy', optimizer="SGD")
        D_compose_G.compile(loss='binary_crossentropy', optimizer=G_optimizer)
        # release the discriminator to be trainable and complile it
        D.trainable = True
        D.compile(loss='binary_crossentropy', optimizer=D_optimizer)
        # start training epochs
        # record discriminator loss D_loss and generator loss G_loss as two arrays to return
        D_loss_record = []
        G_loss_record = []
        for epoch in range(num_epoch):
            # iteration within each epoch
            for index in range(num_iter):
                ##### train discriminator D #####
                # generate Z_1,...,Z_{BATCH_SIZE}
                noise = self.noise_generator(BATCH_SIZE)
                # take a batch of training inputs as real data
                if sampling == 'deterministic_sweep':
                    real_data_batch = train_inputs[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                elif sampling == 'random_uniform':
                    batch_labels = np.random.choice(len(train_inputs), BATCH_SIZE).astype(int)
                    real_data_batch = np.array(train_inputs)[batch_labels]  
                else:
                    print('Class GAN: sampling method error! Input deterministic_sweep or random_uniform \n')
                    real_data_batch = None
                # generator generates a batch of generated data
                # verbose=1 means that we output log
                generated_data_batch = G.predict(noise, verbose=0)
                # concatenate real and generated data batches, real first and then generated follows
                X = np.concatenate((real_data_batch, generated_data_batch))
                # create labels classifying real and generated data
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                # train the discriminator using (X, y) and update the hyperparameters of D
                D_loss = D.train_on_batch(X, y)
                print("epoch %d maximum iteration %d batch number %d discriminator_loss : %f" % (epoch, num_iter, index, D_loss))
                D_loss_record.append(D_loss)

                ##### train generator G #####
                # generate Z_1,...,Z_{BATCH_SIZE}
                noise = self.noise_generator(BATCH_SIZE)
                # lock the discriminator D as not trainable
                D.trainable = False
                # given the discriminator, train the generator G
                G_loss = D_compose_G.train_on_batch(noise, [1] * BATCH_SIZE)
                # unlock the discriminator D as trainable
                D.trainable = True
                print("epoch %d maximum iteration %d batch number %d generator_loss : %f" % (epoch, num_iter, index, G_loss))                                
                G_loss_record.append(G_loss)

        # save the G and D weights
        G.save_weights('generator', True)
        D.save_weights('discriminator', True)

        return D_loss_record, G_loss_record
    

    # Generate samples using the trained GAN generator
    # num is the total number of i.i.d samples being generated
    def generate(self, num):
        # load the trained weights 
        G = self.G
        G.compile(loss='binary_crossentropy', optimizer="SGD")
        G.load_weights('generator')
        D = self.D
        D.compile(loss='binary_crossentropy', optimizer="SGD")
        D.load_weights('discriminator')
        # generate noise, i.e. the hidden variables Z
        noise = self.noise_generator(num)
        # generate samples using the trained GAN generator
        # verbose=1 means that we output log
        generated_samples = G.predict(noise, verbose=0)
        # output the generated samples
        return generated_samples
