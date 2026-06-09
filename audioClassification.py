import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)

DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md')&(commands!='.DS_Store')]
print('Commands:', commands)

train, valid = tf.keras.utils.audio_dataset_from_directory(
  directory=data_dir,
  batch_size=64,
  validation_split=0.2,
  seed=0,
  output_sequence_length=16000,
  subset='both')

label_names=np.array(train.class_names)
print("label names:", label_names)
train.element_spec

def squeeze(audio,labels):
  audio=tf.squeeze(audio,axis=-1)
  return audio, labels
train=train.map(squeeze,tf.data.AUTOTUNE)
valid- valid.map(squeeze,tf.data.AUTOTUNE)

test=valid.shard(num_shards=2, index=0)
valid=valid.shard(num_shards=2, index=1)

for ex_audio, ex_labels in train.take(1):
  print(ex_audio.shape)
  print(ex_labels.shape)

label_names[[1,1,3,0]]
plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
  plt.subplot(rows, cols, i+1)
  audio_signal = ex_audio[i]
  plt.plot(audio_signal)
  plt.title(label_names[ex_labels[i]])
  plt.yticks(np.arange(-1.2, 1.2, 0.2))
  plt.ylim([-1.1, 1.1])

def spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

for i in range(3):
  label=label_names[ex_labels[i]]
  waveform=ex_audio[i]
  spectro=spectrogram(waveform)
  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectro.shape)
  print('Audio playback')
  display.display(display.Audio(waveform, rate=16000))
