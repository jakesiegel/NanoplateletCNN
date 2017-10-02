# CNN for Quantum Coherence Spectral Analysis
# author: Jake Siegel

import numpy as np
from numpy import array
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
import h5py
import csv

batch_size = 30
epochs = 200
N = 324
num_classes=9

# Define Model Architecture
# Inspired by Tuccillo et al.  arXiv: 1701.05917v1 [astro-ph.IM]
# Galaxy Morphology CNN 

model = Sequential()
model.add(Convolution2D(32,(3,3), activation='relu', input_shape=(120,120,3)))
model.add(Convolution2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(128,(2,2), activation='relu'))
model.add(Convolution2D(128,(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

# Compile the model
# note: these are the default sdg parameters to begin with
sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay= 0.0, nesterov=False)
model.compile(loss='mean_squared_error',
			optimizer=sgd,
			metrics=['accuracy'])
			
# need to load in the data
# x_train is built from adding all 3 spectra into a single "RGB"
def generate_training_data(batch_size):	
	x_train = np.zeros((batch_size,120,120,3),dtype=float)
	y_train = np.zeros((batch_size,9),dtype=int)
	while True:
		for k in xrange(0,batch_size):
			with h5py.File('CdSeSim'+str(random.randint(1,N))+'.hdf5','r') as f:
				grp='Sim'+str(random.randint(1,28)) # reserve Sim1 and Sim30 for validation
				tS1 = np.absolute(f[grp]['S1'][:])
				tS2 = np.absolute(f[grp]['S2'][:])
				tS3 = np.absolute(f[grp]['S3'][:])
				x_train[k,:,:,:]=np.stack([tS1,tS2,tS3],axis=2)
# 				Build classes from metadata
# 				currently set classes to model, n_coup = 2*3
				if f[grp].attrs['model']=='single' and f[grp].attrs['n_coup']==1:
					y_train[k] = [1,0,0,0,0,0,0,0,0]
				elif f[grp].attrs['model']=='single' and f[grp].attrs['n_coup']==2:
					y_train[k] = [0,1,0,0,0,0,0,0,0]
				elif f[grp].attrs['model']=='single' and f[grp].attrs['n_coup']==3:
					y_train[k] = [0,0,1,0,0,0,0,0,0]
				elif f[grp].attrs['model']=='uncoupled' and f[grp].attrs['n_coup']==1:
					y_train[k] = [0,0,0,1,0,0,0,0,0]
				elif f[grp].attrs['model']=='uncoupled' and f[grp].attrs['n_coup']==2:
					y_train[k] = [0,0,0,0,1,0,0,0,0]
				elif f[grp].attrs['model']=='uncoupled' and f[grp].attrs['n_coup']==3:
					y_train[k] = [0,0,0,0,0,1,0,0,0]
				elif f[grp].attrs['model']=='coupled' and f[grp].attrs['n_coup']==1:
					y_train[k] = [0,0,0,0,0,0,1,0,0]
				elif f[grp].attrs['model']=='coupled' and f[grp].attrs['n_coup']==2:
					y_train[k] = [0,0,0,0,0,0,0,1,0]
				elif f[grp].attrs['model']=='coupled' and f[grp].attrs['n_coup']==3:
					y_train[k] = [0,0,0,0,0,0,0,0,1]
			yield (x_train,y_train)
			
def generate_validation_data(batch_size):
	x_test = np.zeros((batch_size,120,120,3),dtype=float)
	y_test = np.zeros((batch_size,9),dtype=int)
	while True:
		for k in xrange(0,batch_size):
			with h5py.File('CdSeSim'+str(random.randint(1,N))+'.hdf5','r') as f:
				grp='Sim'+str(random.choice([0,29])) # reserve Sim1 and Sim30 for validation
				tS1 = np.absolute(f[grp]['S1'][:])
				tS2 = np.absolute(f[grp]['S2'][:])
				tS3 = np.absolute(f[grp]['S3'][:])
				x_test[k,:,:,:] = np.stack([tS1,tS2,tS3],axis=2)
# 				Build classes from metadata
# 				currently set classes to model, n_coup = 2*3
				if f[grp].attrs['model']=='single' and f[grp].attrs['n_coup']==1:
					y_test[k] = [1,0,0,0,0,0,0,0,0]
				elif f[grp].attrs['model']=='single' and f[grp].attrs['n_coup']==2:
					y_test[k] = [0,1,0,0,0,0,0,0,0]
				elif f[grp].attrs['model']=='single' and f[grp].attrs['n_coup']==3:
					y_test[k] = [0,0,1,0,0,0,0,0,0]
				elif f[grp].attrs['model']=='uncoupled' and f[grp].attrs['n_coup']==1:
					y_test[k] = [0,0,0,1,0,0,0,0,0]
				elif f[grp].attrs['model']=='uncoupled' and f[grp].attrs['n_coup']==2:
					y_test[k] = [0,0,0,0,1,0,0,0,0]
				elif f[grp].attrs['model']=='uncoupled' and f[grp].attrs['n_coup']==3:
					y_test[k] = [0,0,0,0,0,1,0,0,0]
				elif f[grp].attrs['model']=='coupled' and f[grp].attrs['n_coup']==1:
					y_test[k] = [0,0,0,0,0,0,1,0,0]
				elif f[grp].attrs['model']=='coupled' and f[grp].attrs['n_coup']==2:
					y_test[k] = [0,0,0,0,0,0,0,1,0]
				elif f[grp].attrs['model']=='coupled' and f[grp].attrs['n_coup']==3:
					y_test[k] = [0,0,0,0,0,0,0,0,1]
		yield (x_test,y_test)
					
training_generator=generate_training_data(batch_size)
validation_generator=generate_validation_data(batch_size)

model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
					steps_per_epoch=302, epochs=epochs,
					validation_steps = 21)

# steps per epoch should be N_samples in training set / batchsize = 302.4
# validation steps should be N_samples in validation set / batchsize = 21.6

# exp_data: load scans
with open('ExpS1.csv','rb') as csvfile:
    tS1 = csv.reader(csvfile)
    tempS1=list(tS1)
    S1=np.array(tempS1)

with open('ExpS2.csv','rb') as csvfile:
    tS2 = csv.reader(csvfile)
    tempS2=list(tS2)
    S2=np.array(tempS2)

with open('ExpS3.csv','rb') as csvfile:
    tS3 = csv.reader(csvfile)
    tempS3=list(tS3)
    S3=np.array(tempS3)

ExpS = np.empty((1,120,120,3))
ExpS = np.stack([S1,S2,S3],axis=2).reshape(1,120,120,3)
predictions = model.predict_classes(ExpS)
print(predictions)

model.save('CdSe_trained_Model_ncoup.hdf5')
