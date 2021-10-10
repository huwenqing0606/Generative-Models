"""
The VAE training encapsulated in a class
Environment: Keras 2.3.1 and Tensorflow 1.14
Reference: Kingma, D.P. and Welling, M., Auto-Encoding Variational Bayes, arXiv:1312.6114, Dec. 2013.
Author: Wenqing Hu (Missouri S&T)
"""

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np


class VAE():
    # constructor function
    def __init__(self,
                 encoder_design,    # the design layer of the encoder
                 decoder_model,     # the network architechture of the decoder
                 noise_dim          # the latent noise dimension
                ):
        self.encoder_design = encoder_design
        self.decoder_model = decoder_model
        self.noise_dim = noise_dim


    # the function for reparametrization layer
    # turning [mu, sigma] into Z = mu + eps * exp(sigma/2) where eps ~ N(0, I)
    def reparametrization(self, args):
        mu, sigma = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        eps = K.random_normal(shape=(batch, dim))
        return mu + K.exp(sigma / 2) * eps


    # the VAE loss function
    # must be defined within the same computational graph as the VAE
    # uses function composition technique
    # recons_loss_type is the type of the reconstruction loss, should be 'eucliden' or 'binary_crossentropy'
    def VAE_loss(self, mu, sigma, recons_loss_type):
        def kl_reconstruction_loss(true, pred):
            # the reconstruction loss
            if recons_loss_type == 'euclidean':
                reconstruction_loss =  K.sum(K.square((K.flatten(true)-K.flatten(pred))), axis=-1)
            elif recons_loss_type == 'binary_crossentropy':
                reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred))
            else:
                print('Error! data_loss_type input euclidean or binarycrossentropy \n')
                reconstruction_loss = None
            # the KL divergence loss
            kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            # total loss = %50 reconstruction loss + 50% KL divergence loss
            return K.mean(reconstruction_loss + kl_loss)
        return kl_reconstruction_loss

    
    # build and train the VAE
    # Input Variables:
    #   train_inputs    : the training input data, should be an array where each element is an input array, i.e. array([input, input,...])
    #   optimizer       : the optimizer for VAE network
    #   recons_loss_type: the type of loss used for the reconstruction loss in VAE loss function, should be 'eucliden' or 'binary_crossentropy'
    #   BATCH_SIZE      : the sampling batch size
    #   sampling        : the method of sampling batches, choose 'deterministic_sweep' or 'random_uniform'
    #   num_epoch       : the number of epochs in training
    #   num_iter        : the number of iterations in each epoch
    def train(self, train_inputs, optimizer, recons_loss_type, BATCH_SIZE, sampling, num_epoch, num_iter):
        ### encoder design layer, input data, output a tensor
        # input layer
        i = Input(shape=np.shape(train_inputs[0]), name='encoder_input')
        # encoder design layer
        x = self.encoder_design(i)
        ### encoder [mu, sigma] layer and reparametrization layer
        # a fully connected layer for mu
        mu = Dense(self.noise_dim, name='latent_mu')(x)
        # a fully connected layer for sigma
        sigma = Dense(self.noise_dim, name='latent_sigma')(x)
        # the reparametrization layer Z = mu + eps*exp(sigma/2) where eps ~ N(0, I) 
        Z = Lambda(self.reparametrization, output_shape=(self.noise_dim, ), name='Z')([mu, sigma])
        # build the encoder
        encoder = Model(i, [mu, sigma, Z], name='encoder')
        ### decoder layer
        # decoder takes input Z and output the generated data 
        # build the decoder
        decoder = self.decoder_model(self.noise_dim)
        ### the whold vae 
        vae_outputs = decoder(encoder(i)[2])
        vae = Model(i, vae_outputs, name='vae')
        ### compile VAE
        vae.compile(optimizer=optimizer, loss=self.VAE_loss(mu, sigma, recons_loss_type))
        ### train VAE
        vae_loss_record=[]
        for epoch in range(num_epoch):
            # iteration within each epoch
            for index in range(num_iter):
                # take a batch of training inputs as real data
                if sampling == 'deterministic_sweep':
                    real_data_batch = train_inputs[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                elif sampling == 'random_uniform':
                    batch_labels = np.random.choice(len(train_inputs), BATCH_SIZE).astype(int)
                    real_data_batch = np.array(train_inputs)[batch_labels]  
                else:
                    print('Class VAE: sampling method error! Input deterministic_sweep or random_uniform \n')
                    real_data_batch = None
                # train
                loss = vae.train_on_batch(real_data_batch, real_data_batch)
                # print loss
                print("epoch %d maximum iteration %d batch number %d vae_loss : %f" % (epoch, num_iter, index, loss))            
            vae_loss_record.append(loss)
            # save the encoder and decoder weights
            encoder.save_weights('encoder_weights', True)
            decoder.save_weights('decoder_weights', True)
        return encoder, decoder, vae_loss_record


    # generate new data from input latent noise Z
    # num is the total number of i.i.d samples being generated
    # data_shape is the output data shape, should be np.shape(train_inputs[0])
    def generate(self, decoder, num, data_shape):
        decoder.load_weights('decoder_weights')
        generated_data = []
        for index in range(num):
            # sample the latent variale Z ~ N(0,I)
            Z_sample = np.random.normal(0, 1, size=self.noise_dim)
            # use the trained decoder, get the decoded data
            data_decoded = decoder.predict(np.array([Z_sample]))
            data=data_decoded[0].reshape(data_shape)
            # append to the generated data and return
            generated_data.append(data)
        generated_data=np.array(generated_data)
        return generated_data