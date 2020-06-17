import array
import os
from pydub import AudioSegment
from pydub.playback import play
import numpy as np 
import tensorflow as tf
from tensorflow.keras import regularizers
import re
import functools
print = functools.partial(print, flush=True)
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import save
from numpy import load
import librosa, librosa.display


class_names = ["kick","snare","clap","hihat"] 

train_path_list = ["train_kicks\\","train_snares\\","train_claps\\","train_hats\\"]
test_path_list = ["test_kicks\\","test_snares\\","test_claps\\","test_hats\\"]

arr = []
for a in train_path_list:
	for i in os.listdir(a):
		arr.append(a  + i)


print(len(arr))

sample_list = []

n_fft = 2048
hop_length = 512


def pitchSample(octaves,sound):
	new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
	hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
	hipitch_sound = hipitch_sound.set_frame_rate(44100)
	return hipitch_sound

def bassBoostSample(cutoff,sound):
	#no actually parametric eq so i layer a lp over
	print("cutoff: ",cutoff*10)
	lowpassed = AudioSegment.low_pass_filter(sound,(cutoff*10)+100)
	augmented_sound = sound + lowpassed
	return augmented_sound

def ms_samples(sample_length):
	return int((44100 / 1000) * sample_length)


def augmentor(sound,aug):
	if aug == 1:
		pitched_sounds.append(sound)
	else:
		for i in range(aug):
			aug = float(aug)
			spread = ((aug/100) - (aug*2)/100) + (aug/100)*i
			pitched_sounds.append(pitchSample(spread,sound))
			# pitched_sounds.append(bassBoostSample(spread,sound))
	return pitched_sounds


sample_length = 100 #this is ms!
aug = 9



amount_entries = len(arr)*aug
np_mfcc = np.empty((amount_entries, 9, 13))

#if you already have mfcc's saved it will just load them / if you want to create new ones delete them from the file
if 'np_mfcc.npy' in os.listdir():
	print('loading saved data')
	np_mfcc = load('np_mfcc.npy')
	sample_list = load('sample_list.npy')
else:
	for i in range(len(arr)): #create mfcc

		# cant figure out how to play 32bit file
		sound = AudioSegment.from_file(arr[i], format="wav", channels=1)
		sound = sound.set_channels(1)
		
		pitched_sounds = []
		augmentor(sound,aug)


		for x in range(len(pitched_sounds)):

			
			if re.search("kicks",arr[i]):
				sample_list.append(0)
			elif re.search("snares",arr[i]):
				sample_list.append(1)
			elif re.search("clap",arr[i]):	
				sample_list.append(2)
			else:	
				sample_list.append(3)


			sound = pitched_sounds[x][:sample_length]

			samples = sound.get_array_of_samples()


			if len(samples) < ms_samples(sample_length):
				padding_samples = ms_samples(sample_length) - len(samples) 
				for dumi in range(padding_samples):
					samples.append(0)


			#turned audio segment into mfcc
			samples = np.array(samples)
			samples = samples.astype(float)
			mfcc = librosa.feature.mfcc(samples, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
			mfcc = mfcc.T
			# librosa.display.specshow(mfcc, sr=44100, hop_length=hop_length)
			# plt.show()
			mfcc = np.expand_dims(mfcc, axis=0)

			# np_mfcc = np.append(np_mfcc, mfcc, axis=2)
			np_mfcc[i*aug+x] = mfcc
				


		if i % 100 == 0:
			print(np.floor((i*100)/len(arr)))
	save('np_mfcc.npy', np_mfcc)
	sample_list = np.array(sample_list)
	save('sample_list.npy', sample_list)

#you need to have a 3d shape for a conv2D layer, 4d if you include batch size
np_mfcc = np.expand_dims(np_mfcc, axis=3)


#shuffle ONCE
seed = 10
np.random.seed(seed)
np.random.shuffle(np_mfcc)
np.random.seed(seed)
np.random.shuffle(sample_list)
np.random.seed()


#new test/validation samples
arr = []
for a in test_path_list:
	for i in os.listdir(a):
		arr.append(a  + i)


test_sample_list = []
# test_np_samples = np.empty((len(arr),ms_samples(sample_length)))
test_np_mfcc = np.empty((len(arr), 9, 13))

for i in range(len(arr)):

	# cant figure out how to play 32bit file
	sound = AudioSegment.from_file(arr[i], format="wav", channels=1)
	sound = sound.set_channels(1)

	sound = sound[:sample_length]
	samples = sound.get_array_of_samples()

	if re.search("kicks",arr[i]):
		test_sample_list.append(0)
	elif re.search("snares",arr[i]):
		test_sample_list.append(1)
	elif re.search("claps",arr[i]):
		test_sample_list.append(2)
	else:
		test_sample_list.append(3)


	if len(samples) < ms_samples(sample_length):
		padding_samples = ms_samples(sample_length) - len(samples) 
		for padno in range(padding_samples):
			samples.append(0)

	# test_np_samples[i] = samples
	samples = np.array(samples)
	samples = samples.astype(float)
	mfcc = librosa.feature.mfcc(samples, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
	mfcc = mfcc.T

	mfcc = np.expand_dims(mfcc, axis=0)
	test_np_mfcc[i] = mfcc


test_sample_list = np.array(test_sample_list)
test_np_mfcc = np.expand_dims(test_np_mfcc, axis=3)
# test_np_samples = test_np_samples.astype(float) / (2**15)


#shuffle ONCE
seed = 10
np.random.seed(seed)
np.random.shuffle(test_sample_list)
np.random.seed(seed)
np.random.shuffle(test_np_mfcc)
np.random.seed(seed)
np.random.shuffle(arr)
np.random.seed()


train_size = int(amount_entries * 0.9)
val_size = amount_entries - train_size

training_ds = tf.data.Dataset.from_tensor_slices((np_mfcc,sample_list))
# dataset = training_ds.shuffle(train_size + val_size)


val_ds = training_ds.skip(train_size).take(val_size)
training_ds = training_ds.take(train_size)

print(val_ds)
print(training_ds)



####################build model##############

# batch_size = np_samples.shape[0]
batch_size = 320
STEPS_PER_EPOCH = train_size//batch_size

def get_callbacks():
  return [
    # tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6),
    # tf.keras.callbacks.TensorBoard(logdir/name),
  ]

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)



inputShape = (9, 13, 1)

model = tf.keras.models.Sequential([


	#1st conv layer
	tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(9, 13, 1), kernel_regularizer=regularizers.l2(0.001)),
	tf.keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'),
	tf.keras.layers.BatchNormalization(),

	#2nd conv layer
	# tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)), #?? does it not need input layers here?
	# tf.keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'),
	# tf.keras.layers.BatchNormalization(),

	# #3rd conv layer
	# tf.keras.layers.Conv2D(32, (2,2), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
	# tf.keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'),
	# tf.keras.layers.BatchNormalization(),

	#flatten & dense & output
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Dense(len(train_path_list))
	])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



history = model.fit(training_ds.shuffle(train_size).batch(batch_size), 	
	epochs=40, 
	validation_data=val_ds.batch(batch_size),
	# validation_steps=val_size, 
	callbacks=get_callbacks(),
	verbose=1,
	# batch_size=32)
	)



model.evaluate(test_np_mfcc, test_sample_list, verbose=2)

#history history plot stuff
	# history_dict = history.history

	# acc = history_dict['accuracy']
	# val_acc = history_dict['val_accuracy']
	# loss = history_dict['loss']
	# val_loss = history_dict['val_loss']

	# epochs = range(1, len(acc) + 1)

	# # "bo" is for "blue dot"
	# plt.plot(epochs, loss, 'bo', label='Training loss')
	# # b is for "solid blue line"
	# plt.plot(epochs, val_loss, 'b', label='Validation loss')
	# plt.title('Training and validation loss')
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.legend()

	# plt.show()


	# plt.clf()   # clear figure

	# plt.plot(epochs, acc, 'bo', label='Training acc')
	# plt.plot(epochs, val_acc, 'b', label='Validation acc')
	# plt.title('Training and validation accuracy')
	# plt.xlabel('Epochs')
	# plt.ylabel('Accuracy')
	# plt.legend(loc='lower right')

	# plt.show()



probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


for i in range(len(test_np_mfcc)):
	result = probability_model(test_np_mfcc[i:i+1]) #because it needs to be a list in a list, [i,4410] just returns a 1d list
	answer = np.argmax(result[0])    
	print(arr[i] + " is a " + class_names[answer])




def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  # plt.imshow(img, cmap=plt.cm.binary)
  plt.plot(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(4))
  plt.yticks([])
  thisplot = plt.bar(range(4), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')    

