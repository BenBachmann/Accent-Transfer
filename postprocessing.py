import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
import numpy as np
from pydub import AudioSegment
import preprocessing
import scipy.io.wavfile as wav
import model

#data = preprocessing.mp3_to_numpy()[0][0]
data = model.transformer()

rate = 44100
scaled = np.int16(data / np.max(np.abs(data)) * 32767)
wav.write('test.wav', rate, scaled)