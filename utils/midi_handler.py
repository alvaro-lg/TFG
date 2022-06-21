import math
import numpy as np
import mido
from matplotlib import pyplot as plt

# Constantes relativas al estádar MIDI
DEFAULT_SR = 500000

class Midi_handler:

    def __init__(self,  sampling_rate, dir_path, n_notes):
        self.sampling_rate = sampling_rate
        self.dir_path = dir_path
        self.n_notes = n_notes

    '''
        Los mensajes son eventos que, dependiendo de su tipo nos indican información acerca de 
        lo que sucede en la pista. En este método nos quedamos sólo con la información que nos 
        interesa dependiendo del tipo de mensaje.
    '''
    def msg2dict(self, msg):
        # Declaración de variables
        result = dict()
        key_on = bool()

        # Para saber si el mensaje nos indica que una nota esta pulsada en ese momento, o si
        # el mensaje no se refiere a una nota
        if msg.type == 'note_on':
            key_on = True
        elif msg.type == 'note_off':
            key_on = False
        else:
            key_on = None

        # Obtenemos la información relativa al tiempo
        result['time'] = msg.time

        # En caso de que se trate de una nota, retornamos los atributos que nos interesan
        if key_on is not None:
            result['note'] = msg.note
            result['velocity'] = msg.velocity

        return (result, key_on)

    '''
        Actualiza el valor de una nota en un nuevo estado para unos valores de velocidad
        faciltados de entrada.
    '''
    def switch_note(self, last_state, note, velocity, key_on):
        # Si se trata del primer estado de la pista, creamos un array de ceros vacio
        if last_state is None:
            result = np.zeros((self.n_notes), dtype=int)
        else:
            result = last_state.copy()

        # Si la nota esta dentro de los margenes de las que estan en los valores del teclado
        # del piano y se pulsa, actualizamos su velocidad
        if 21 <= note <= 108:
            if key_on:
                result[note - 21] = velocity
            else:
                result[note - 21] = 0

        return result

    '''
        Devuelve un nuevo estado de acuerdo a un nuevo mensaje. Entendemos por estado, como
        un vector (array) donde cada componente representa el valor de velocidad de una de
        las teclas del piano en un instante de tiempo determinado.
    '''
    def get_new_state(self, new_msg, last_state):
        # Obtenemos los distintos valores y parámetros del mensaje en cuestión
        new_msg, key_on = self.msg2dict(new_msg)

        if key_on is not None:  # Si el mensaje se refiere a un cambio en una nota
            new_state = self.switch_note(last_state, new_msg['note'], new_msg['velocity'], key_on)
        else:  # En caso contrario el estado se mantiene inmutable
            new_state = last_state

        return (new_state, new_msg['time'])

    '''
        Transformamos una pista (compuesta de mensajes) en un array bidimensional de estados.
        Hay que convertir los mesajes a estados (np.array(s)).
    '''
    def track2seq(self, track):
        # Lista donde almacenaremos los distintos estados
        result = []

        # Obtenemos los primeros valores de estado y tiempo
        last_state, last_time = self.get_new_state(track[0], np.zeros((self.n_notes), dtype=int))

        for i in range(1, len(track)):
            new_state, new_time = self.get_new_state(track[i], last_state)

            # Añadimos tantas "rodajas" temporales con el ultimo estado como tiempo haya pasado
            # desde la ultima pulsacion
            if new_time > 0:
                result += [last_state] * new_time
            last_state, last_time = new_state, new_time

        return np.array(result)

    '''
        Devuelve un array con los valores del pianoroll de todas las pistas del midi.
    '''
    def mid2array(self, mid):
        # Lista donde almacenaremos los valores de las distintas pistas
        all_arys = []

        # Calculamos cual va a ser la longitud de nuestro array
        tracks_len = [len(tr) for tr in mid.tracks]

        # Obtenemos los np.array(s) de cada pista
        for i in range(1, len(mid.tracks)):
            all_arys.append(self.track2seq(mid.tracks[i]))

        # Convertirmos a np.array y nos quedamos con el valor máximo de una nota de entre todas las pistas
        all_arys = np.array(all_arys)
        all_arys = all_arys.max(axis=0)

        return all_arys

    '''
        Devuelve el pianorroll pasado como argumento resampleado al ratio que se proporiciona.
    '''
    def resample_pianoroll(self, piano_roll, old_sampling_rate, new_sampling_rate):
        step = old_sampling_rate / new_sampling_rate

        result = []

        for i in np.arange(0.0, len(piano_roll), step):
            result.append(np.mean(np.array(piano_roll[int(i):math.ceil(i + step)]), axis=0))

        return np.array(result)

    '''
        Analiza los mensajes de las pistas de un determinado archivo midi que se pasa como
        argumento de entrada para calcular su sampling rate y lo devuelve.
    '''
    def get_sampling_rate(self, midi):
        tpb = midi.ticks_per_beat
        t_ms = DEFAULT_SR  # Valor por defecto en el formato midi
        track = midi.tracks[0]  # Pista que contiene metapaámetros del archivo

        for msg in track:
            if msg.tempo is not None:
                t_ms = msg.tempo
                break

        t_s = (10 ** 6) / t_ms
        return tpb * t_s

    '''
        Plotea un pianoroll pasado como argumento de entrada.
    '''
    def plot_pianoroll(self, pianoroll, title=None):
        plt.figure(figsize=(16, 6), dpi=200)
        plt.plot(range(pianoroll.shape[0]), np.multiply(np.where(pianoroll > 0, 1, 0), range(1, self.n_notes + 1)), \
                 marker='.', markersize=1, linestyle='', color='c')
        plt.xlim(0, pianoroll.shape[0] * 1.05)
        plt.ylim(0, self.n_notes * 1.05)
        if title is not None: plt.title(title)
        plt.show()

    '''
        Toma un de archivo MIDI y devuelve sy pianorolls. Tiene argumentos opcionales para especificar 
        el directorio padre desde el que ha de leerse el archivo MIDI en cuestión y la frecuencia
        de muestreo a la que ha de devolverse.
    '''
    def vectorize_midi(self, file):
        midi = mido.MidiFile(self.dir_path + file, clip=True)
        pianoroll = self.mid2array(midi)
        return self.resample_pianoroll(pianoroll, self.get_sampling_rate(midi), self.sampling_rate)

    def vectorize_midi_generator(self, file):
        midi = mido.MidiFile(self.dir_path + file, clip=True)
        pianoroll = self.mid2array(midi)

        for item in self.resample_pianoroll(pianoroll, self.get_sampling_rate(midi), self.sampling_rate):
            yield item