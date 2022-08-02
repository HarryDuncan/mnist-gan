
#Importing required library.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Input,Conv2D,Conv2DTranspose,Flatten,Reshape
from keras.models import Model,Sequential
from keras.datasets import mnist
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras import layers
from tqdm import tqdm

#Loading and normalizing the data.
(X_train,y_train),(X_test,y_test)=mnist.load_data() 
X_train=X_train.astype('float32')
X_train=np.expand_dims(X_train,-1)
X_train=X_train/255 

def define_generator(latent_dim):

	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def define_gan(g_model, d_model):
    d_model.trainable=False
    #Input tensor taking 100 dim noise vec.
    inp=Input(shape=(100,))
    #Passing it through generator.
    x=g_model(inp)
    #Feeding the generator output to discriminator.
    x=d_model(x)
    model=Model(inputs=inp,outputs=x)
  
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# size of the latent space
latent_dim = 100
# create the discriminator
d = define_discriminator()
# create the generator
g= define_generator(latent_dim)
# create the gan
gan = define_gan(g, d)

#Train the model.

#Basic parameters.
batch_size=256
half_batch=128
epochs=100
latent_dim=100
batch_per_epoch=X_train.shape[0]//batch_size




for epoch in range(epochs):
  d1_loss=[]
  d2_loss=[]
  g_loss=[]
  for _ in tqdm(range(batch_per_epoch)):
    X_real=X_train[np.random.randint(0,X_train.shape[0],half_batch)] #Selecting n=batch_size random images from training data.
    y_real=np.ones((half_batch,1)) #Setting label to 1 as they are real samples 

    X_fake=np.random.randn(latent_dim*half_batch) #Creating batch_size no of latent vector of dim 100.
    X_fake=X_fake.reshape(half_batch,latent_dim)
    #Generating fake image on which we want to train generator.
    x_fake_img=g.predict(X_fake)
    y_fake=np.zeros((half_batch,1)) #Setting the label to 0 as they are fake samples.

    #Train the generator with both real and fake samples seperately.
    dLReal,_=d.train_on_batch(X_real,y_real)
    d1_loss.append(dLReal)
    
    dLFake,_=d.train_on_batch(x_fake_img,y_fake)
    d2_loss.append(dLFake)

    #Now lets train generator.For training generator we have to train it through discriminator.
    #Sample batch_size no of latent vector.
    X_fake_gan=np.random.randn(latent_dim*batch_size) #Creating batch_size no of latent vector of dim 100.
    X_fake_gan=X_fake_gan.reshape(batch_size,latent_dim)
    """ Here we set label=1. Its more like telling the generator to update itself in 
        such a way that discriminator classifies the sample as 1 that is as real 
        sample.
    """
    y_gan=np.ones((batch_size,1))
    g_l=gan.train_on_batch(X_fake_gan,y_gan)
  
  #Plot 10 generated images per 5 epochs.
  if epoch%5==0:
    l_vec=np.random.randn(latent_dim*10) 
    l_vec=l_vec.reshape(10,latent_dim)
    gen_images=g.predict(l_vec)
    
    count=0
    fig,ax=plt.subplots(2,5)
    for i in range(2):
      for j in range(5):
        ax[i][j].axis('off')
        ax[i][j].imshow(np.squeeze(gen_images[count],axis=-1),cmap='gray')
        count+=1
    fig.savefig(f'gen_images{epoch}.jpg')
    g_loss.append(g_l)
  print(f'\nEpoch={epoch} DLReal={sum(d1_loss)/(batch_per_epoch)} DLFake={sum(d2_loss)/(batch_per_epoch)} GAN_loss={sum(g_loss)/batch_per_epoch}' )


def generate_and_visualize():
    l_vec=np.random.randn(latent_dim*10) 
    l_vec=l_vec.reshape(10,latent_dim)
    gen_images=g.predict(l_vec)
    
    count=0
    fig,ax=plt.subplots(2,5)
    for i in range(2):
      for j in range(5):
        ax[i][j].axis('off')
        ax[i][j].imshow(np.squeeze(gen_images[count],axis=-1),cmap='gray')
        count+=1
    plt.show()

generate_and_visualize()