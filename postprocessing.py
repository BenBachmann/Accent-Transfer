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

test_model = transformer.Transformer(num_layers=3)
print("A")
test_model.build(input_shape=(2, 11114, 1))
print("B")
test_model.load_weights('transformer_weights_v1.h5')
print("C")
weights = test_model.get_weights()
print("D")

all_data = preprocessing.unpickle_file("./chinese.pickle")
print("E")

input_data = all_data[18]
input_data = np.expand_dims(input_data, axis=0)
print("F")
output_data = test_model.predict(input_data)
print("G")


rate = 44100
scaled = np.int16(output_data / np.max(np.abs(output_data)) * 32767)
wav.write('transformer1.wav', rate, scaled)