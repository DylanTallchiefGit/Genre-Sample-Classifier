import os
import librosa, librosa.display
import math
from pydub import AudioSegment
# from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt 
from numpy import asarray
from numpy import save
from numpy import load
import functools
print = functools.partial(print, flush=True) #this actually does make it a lot slower
import tensorflow as tf
from tensorflow.keras import regularizers


dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(dir_path)
			

# signal, sr = librosa.load(file_path, sr=SAMPLE_RATE) # this is the soundfile method, which doesnt work with mp3/webm
#check the samplerates = all webm youtube downloads have 48khz
#pydub can open mp3/webm files and convert to samples/numpy arrays. thank you pydub
#takes about 14 seconds to open 1hr long webm file, optimal way would be to just open chunks/buffers of it but oh well



def plot_history(history):
	fig, axs = plt.subplots(2)

	#create accuracy subplot
	axs[0].plot(history.history["accuracy"], label="train accuracy")
	axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
	axs[0].set_ylabel("accuracy")
	axs[0].legend(loc="lower right")
	axs[0].set_title("Accuracy")

	#create loss subplot
	axs[1].plot(history.history["loss"], label="train loss")
	axs[1].plot(history.history["val_loss"], label="validation loss")
	axs[1].set_ylabel("loss")
	axs[1].set_ylabel("Epoch")
	axs[1].legend(loc="upper right")
	axs[1].set_title("Loss")

	plt.show()



mfcc_counter = 0
n_fft = 2048
hop_length = 512
mlt = 2 #multiplier (yeah its weird)
chunk = 1e3 * mlt  #1 second # 1e3 = 1000


if 'mfcc_features.npy' in os.listdir():
	print('loading saved data')
	mfcc_features = load('mfcc_features.npy')
	mfcc_labels = load('mfcc_labels.npy')
else:

	mfcc_features = np.empty((0,94*mlt,13))
	mfcc_labels = []

	for i, (root, dirs, filenames) in enumerate(os.walk(dir_path)):
		for name in filenames:
			
			if name.endswith('.webm'):

				file_path = os.path.join(root, name)
				print(file_path, i, file_path.split('\\')[-2])
				sound = AudioSegment.from_file(file_path)
				sound = sound.set_channels(1)
				lenght_audio = len(sound)

				number_mfccs = math.floor(lenght_audio/chunk)
				print(lenght_audio/6e4,number_mfccs) #length in min, number chunks

				mfcc_features_add = np.empty((number_mfccs,94*mlt,13))
				mfcc_features = np.append(mfcc_features, mfcc_features_add, axis=0)

				print(mfcc_features_add.shape)
				print(mfcc_features.shape)

				for x in range(number_mfccs):
					cut = sound[(x*chunk):(x*chunk)+chunk]
					samples = cut.get_array_of_samples()
					samples = np.array(samples)
					samples = samples.astype(float)
		
					# librosa.display.waveplot(samples, sr=sound.frame_rate)
					# plt.xlabel('Time')
					# plt.ylabel('Amplitude')
					# plt.show()

					mfcc = librosa.feature.mfcc(samples, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
					mfcc = mfcc.T
					# librosa.display.specshow(mfcc, sr=44100, hop_length=hop_length)
					# plt.show()


					mfcc_features[mfcc_counter] = mfcc
					mfcc_labels.append(i-1)

					mfcc_counter += 1 

				sound = None

	mfcc_features = np.expand_dims(mfcc_features, axis=3)
	save('mfcc_features.npy', mfcc_features)
	mfcc_labels = np.array(mfcc_labels)
	save('mfcc_labels.npy', mfcc_labels)

print(mfcc_features.shape)

output_neurons = mfcc_labels[-1] + 1
print('output_neurons: ',output_neurons)


#normalize??
mean = np.mean(mfcc_features, axis=0)
std = np.std(mfcc_features, axis=0)
mfcc_features = (mfcc_features - mean) / std



#shuffle dataset ONCE
seed = 10
np.random.seed(seed)
np.random.shuffle(mfcc_features)
np.random.seed(seed)
np.random.shuffle(mfcc_labels)
np.random.seed()



train_size = int(len(mfcc_labels) * 0.9)
val_size = len(mfcc_labels) - train_size

print(train_size)
print(val_size)

training_ds = tf.data.Dataset.from_tensor_slices((mfcc_features,mfcc_labels))
# training_ds = training_ds.shuffle(train_size + val_size)


val_ds = training_ds.skip(train_size).take(val_size)
training_ds = training_ds.take(train_size)

print(val_ds)
print(training_ds)


#these two do the same thing, neat
# print(next(iter(training_ds.take(1)))[1])
# for item in training_ds.take(1):
# 	print(item[1])




####################build model##############

# batch_size = np_samples.shape[0]
batch_size = 32 * 1
STEPS_PER_EPOCH = train_size//batch_size
checkpoint_path = 'savedweights_1/chkp-{val_accuracy}.ckpt'

def get_callbacks():
  return [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10), #this will stop the training if there is no improvement since the last 10 epochs
    # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, period=1) #unmute if you want to save ur weights during training
  ]

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)


inputShape = (94*mlt, 13, 1)

model = tf.keras.models.Sequential([


	#1st conv layer
	tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(94*mlt, 13, 1), kernel_regularizer=regularizers.l2(0.001)),
	tf.keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'),
	tf.keras.layers.BatchNormalization(),

	#2nd conv layer
	tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)), #?? does it not need input layers here?
	tf.keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'),
	tf.keras.layers.BatchNormalization(),

	# #3rd conv layer
	tf.keras.layers.Conv2D(32, (2,2), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
	tf.keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'),
	tf.keras.layers.BatchNormalization(),

	#flatten & dense & output
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Dense(output_neurons)
	])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              # loss=loss_fn,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# train
history = model.fit(training_ds.batch(batch_size), 	
	epochs=40, 
	validation_data=val_ds.shuffle(val_size).batch(batch_size),
	callbacks=get_callbacks(),
	verbose=1,
	)


model.evaluate(val_ds.batch(batch_size), verbose=2)


#plot accuracy and error over the epochs
plot_history(history)


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


test_file = 'skrillex scary.webm'
sound = AudioSegment.from_file(test_file)
sound = sound.set_channels(1)
lenght_audio = len(sound)

number_mfccs = math.floor(lenght_audio/chunk)
print(lenght_audio/6e4,number_mfccs) #length in min, number chunks

test_mfccs = np.empty((number_mfccs,94*mlt,13))

for x in range(number_mfccs):
	cut = sound[(x*chunk):(x*chunk)+chunk]
	samples = cut.get_array_of_samples()
	samples = np.array(samples)
	samples = samples.astype(float)
	mfcc = librosa.feature.mfcc(samples, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
	mfcc = mfcc.T
	# librosa.display.specshow(mfcc, sr=44100, hop_length=hop_length)
	# plt.show()
	test_mfccs[x] = mfcc

# test_mfccs = np.expand_dims(test_mfccs, axis=0)
test_mfccs = np.expand_dims(test_mfccs, axis=3)
print(test_mfccs.shape)

#normalize
mean = np.mean(test_mfccs, axis=0)
std = np.std(test_mfccs, axis=0)
test_mfccs = (test_mfccs - mean) / std



class_names = ['ambient', 'classical', 'dubstep', 'hardstyle', 'jazz', 'trance']
prob_results = []

for i in range(len(test_mfccs)):
	result = probability_model(test_mfccs[i:i+1]) #because it needs to be a list in a list, [i,4410] just returns a 1d list
	answer = np.argmax(result[0])
	prob_results.append(answer) 
	print("Skrillex is a " + class_names[answer])