import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

signal, sr = librosa.load('/Users/alvarolopezgarcia/Documents/Documentos Universidad/4o Curso/2o Cuatrimestre/Trabajo de Fin de Grado - Archivos Locales/Ilustraciones/Espectrogramas/Paganini.wav')

signal = signal[0:500000]

# this is the number of samples in a window per fft
n_fft = 2048
# The amount of samples we are shifting after each fft
hop_length = 512

# Short-time Fourier Transformation on our audio data
audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
# gathering the absolute values for all values in our audio_stft
spectrogram = np.abs(audio_stft)
# Converting the amplitude to decibels
log_spectro = librosa.amplitude_to_db(spectrogram)
# Plotting the short-time Fourier Transformation
plt.figure(figsize=(7, 4))
# Using librosa.display.specshow() to create our spectrogram
librosa.display.specshow(log_spectro, sr=sr, x_axis='time', y_axis='hz', hop_length=hop_length, cmap='magma')
plt.tight_layout()
plt.rcParams['text.usetex'] = True
plt.colorbar(label='Decibelios')
plt.title('Espectrograma (dB)', fontdict=dict(size=18))
plt.xlabel('Tiempo (s)', fontdict=dict(size=15))
plt.ylabel('Frecuencia (Hz)', fontdict=dict(size=15))
plt.savefig('imagen.png', dpi=200, bbox_inches='tight')