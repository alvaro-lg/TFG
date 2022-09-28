import librosa
import numpy as np
from matplotlib import pyplot as plt

class Wav_handler:

    def __init__(self, sampling_rate, dir_path):
        self.sampling_rate = sampling_rate
        self.dir_path = dir_path

    '''
        Plotea los valores de la onda del fichero de audio que han pasado como argumento
        de entrada.
    '''
    def plot_wav(self, waves, title=None):
        wave = waves.flatten()
        plt.figure(figsize=(6, 4), dpi=350)
        plt.plot(wave)
        plt.xlim(0, len(wave))
        plt.ylim(min(wave) * 1.05, max(wave) * 1.05)
        if title is not None: plt.title(title)
        plt.savefig('wav.png', bbox_inches='tight', dpi=150)

    '''
        Método que se apoya en la libería librosa para leer un fichero .wav pasado como
        argumento de entrada y devolver los datos correspondientes a los niveles de onda
        como un np.array.
    '''
    def vectorize_wav(self, file):
        data, sr = librosa.load(self.dir_path + file, sr=self.sampling_rate)
        return np.array(data)