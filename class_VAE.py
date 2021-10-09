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
                 train_inputs,
                 noise_dim,
                 data_loss_type,
                 encoder_design,
                 decoder_model
                ):
        self.train_inputs = train_inputs
        self.noise_dim = noise_dim
        self.data_loss_type = data_loss_type
        self.encoder_design = encoder_design
        self.decoder_model = decoder_model


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
    def VAE_loss(self, mu, sigma):
        def kl_reconstruction_loss(true, pred):
            # the reconstruction loss
            if self.data_loss_type == 'euclidean':
                reconstruction_loss =  K.sum(K.square((K.flatten(true)-K.flatten(pred))), axis=-1) 
            elif self.data_loss_type == 'binarycrossentropy':
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
    def train(self, optimizer, BATCH_SIZE, num_epochs, num_iteration):
        ### encoder design layer, input data, output a tensor
        # input layer
        i = Input(shape=np.shape(self.train_inputs[0]), name='encoder_input')
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
        vae.compile(optimizer=optimizer, loss=self.VAE_loss(mu, sigma))
        ### train VAE
        loss_seq=[]
        for epoch in range(num_epochs):
            for index in range(num_iteration):
                # build the training batch
                train_inputs_batch = self.train_inputs[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                # train
                loss = vae.train_on_batch(train_inputs_batch, train_inputs_batch)
                loss_seq.append(loss)
            # save the encoder and decoder weights
            encoder.save_weights('encoder_weights', True)
            decoder.save_weights('decoder_weights', True)
            # print loss
            print("epoch %d maximum iteration %d vae_loss : %f" % (epoch, num_iteration, loss))            
        return encoder, decoder, loss_seq


    # generate new data from input latent noise Z
    def generate(self, decoder, num_samples):
        decoder.load_weights('decoder_weights')
        generated_data = []
        for index in range(num_samples):
            Z_sample = np.random.normal(0,1,size=self.noise_dim)
            data_decoded=decoder.predict(np.array([Z_sample]))
            data=data_decoded[0].reshape(np.shape(self.train_inputs[0]))
            generated_data.append(data)
        generated_data=np.array(generated_data)
        return generated_data