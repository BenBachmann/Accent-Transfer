import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
import numpy as np
from pydub import AudioSegment
import preprocessing
import scipy.io.wavfile as wav
import model
import transformer
import tensorflow as tf

#data = preprocessing.mp3_to_numpy()[0][0]
# data = model.transformer()


all_data, max_len = preprocessing.mp3_to_numpy("chinese", 10)

all_data = tf.keras.preprocessing.sequence.pad_sequences(all_data, maxlen=max_len, padding='post', value=0, dtype='float32')

test_model = transformer.Transformer(num_layers=3)
test_model.build(input_shape=(2, max_len, 1)) # build calls the model.
# print("A")
test_model.load_weights('transformer_weights_v1.h5')
# print("B")
weights = test_model.get_weights()
# print("C")



input_data = all_data[18]
input_data = np.expand_dims(input_data, axis=-1)
input_data = np.expand_dims(input_data, axis=0)
print("F")
# correct up to here.
# input_data has shape (1, 11114, 1)

# error: model.predict sets the first dimension to None
output_data = test_model.predict(input_data, batch_size=1)
print("G")
output_data = np.squeeze(output_data)
print("SHAPE", output_data.shape)


rate = 44100
scaled = np.int16(output_data / np.max(np.abs(output_data)) * 32767)
# print("SCALED", scaled.shape)
wav.write('transformer2.wav', rate, scaled)