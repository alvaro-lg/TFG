#!/usr/bin/env python
# coding: utf-8

# # Interpretación de música de piano usando técnicas de Deep Learning.
# ## Trabajo de Fin de Grado.
# ### Facultad de Ciencias - Universidad de Cantabria.
# **Autor:** Álvaro López García. <br>
# **Tutor:** Cristina Tirnauca.

# In[1]:


'''# Colab-only
from google.colab import drive
drive.mount('/content/drive')
!pip install mido
%cd drive/MyDrive/Colab\ Notebooks/TFG/'''


# In[2]:


# Celda para importar librerías, costantes, etc.
import tensorflow as tf
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import mido
import math
import copy
import gc
import queue
import threading
import time

from matplotlib import pyplot as plt
from IPython import display
from tqdm import tqdm

from IPython.display import display, HTML
display(HTML("<style>.container { width:70% !important; }</style>"))

# Para entrenarlo en mi equipor personal:
PATH = '/Volumes/TheVault/Documentos Mac/Documentos Universidad/4o Curso/2o Cuatrimestre/Trabajo de Fin de Grado/maestro-v3.0.0/'
#PATH = 'maestro-v3.0.0/'


# ### Creación del dataframe

# In[3]:


def get_dataframe(path=PATH + 'maestro-v3.0.0.csv'):
    return pd.read_csv(path)

def set_dataframe(df, path=PATH + 'maestro-v3.0.0.csv'):
    df.to_csv(path, index=False)


# In[4]:


df = get_dataframe('maestro-v3.0.0.csv')
df.head()


# ## Clases, métodos y constantes para el procesado del Dataset

# ### Parámetros del procesado

# In[5]:


#Parámetros de configuración de los archivos de audio
SAMPLING_RATE = 16000
N_NOTES = 88


# ###  Código para el procesado de los datos relativos a los ficheros MIDI (`*.midi -> tf.Dataset`)

# In[6]:


# Constantes relativas al estádar MIDI
DEFAULT_SR = 500000


# In[7]:


from utils.midi_handler import Midi_handler
midi_hdlr = Midi_handler(sampling_rate=SAMPLING_RATE, dir_path=PATH, n_notes=N_NOTES)


# In[8]:


'''# Ploteamos una sección de un pianoroll de ejemplo
midi_hdlr.plot_pianoroll(midi_hdlr.vectorize_midi(df['midi_filename'][0])[0:640000], title=df['midi_filename'][0])'''


# Para los datos relativos a los ficheros `.midi` aplicaremos la normalización **MinMax** donde cada nuevo valor $x'$ para cada ejemplo $x$ vendrá dado por: $$ x' = \frac{x - min}{max - min} ,$$ donde $max$ y $min$ son los valores máximos y mínimos de los valores de entrada. En este caso, dada la definición del estándar MIDI, sabemos que los valores de estas velocidades ya están parametrizados entre $0$ y $127$ por lo que no necesitamos iterar sobre los datos para conocer el máximo y el mínimo. Para ello definiremos las siguientes constantes:

# In[9]:


# Constates para aplicar la normalización a los ficheros MIDI
MIDI_MAX = 127
MIDI_MIN = 0


# ### Código para el procesado de los datos relativos a los ficheros de audio (`*.wav -> tf.Dataset`)

# In[10]:


from utils.wav_handler import Wav_handler
wav_hdlr = Wav_handler(sampling_rate=SAMPLING_RATE, dir_path=PATH)


# In[11]:


'''# Ploteamos una sección de una onda
wav_hdlr.plot_wav(wav_hdlr.vectorize_wav(df['audio_filename'][0])[0:128000], title=df['audio_filename'][0])'''


# Para la compresión de la señal emplearemos una compresión basada en una $\mu$-law.

# In[26]:


# Constates relativas a la compresión con la mu-law
N_BITS = 7
N_LEVELS = 2 ** N_BITS
MU = 256
AMPS = (-1, 1)


# In[27]:


from utils.mu_law_encoder import Mu_law_encoder
mulaw_enc = Mu_law_encoder(mu=MU, n_bits=N_BITS, amps=AMPS)


# ## Creación del modelo
# ![WaveNet_residblock.png](attachment:WaveNet_residblock.png)

# In[28]:


def calculate_receptive_field(n_blocks, n_layers):
    return sum([2 ** i for i in range(n_layers)] * n_blocks) - n_blocks + 1


# In[29]:


# Parámetros que definen la arquitectura del modelo a crear por defecto en las llamadas a get_wavenet()

# Parámetros de la arquitectura interna de la red
N_FILTERS = N_LEVELS
FILTER_WIDTH = 2

# Main wavenet
N_BLOCKS = 3
K_LAYERS = 10

# Context stack
CONTEXT_BLOCKS = 2
CONTEXT_LAYERS = 6

# Números y longitudes de ejemplos de entrada, salida y local conditioning
INPUT_LEN = calculate_receptive_field(N_BLOCKS, K_LAYERS)
N_IN_CHANNELS = N_LEVELS
LC_LEN = 1
N_LC_CHANNELS = N_NOTES

# Dimensionalidad de input/output/local conditioning
INP_SHAPE = (INPUT_LEN, N_IN_CHANNELS)
OUT_SHAPE = INP_SHAPE
LC_SHAPE = (LC_LEN, N_LC_CHANNELS)


# In[30]:


'''
    Método que devuelve el modelo de una WaveNet implementada en tensorflow. Dicha red se construye 
    de acuerdo a los parámetros de entrada.
'''
def get_wavenet(inp_shape=INP_SHAPE, out_shape=OUT_SHAPE, lc_shape=LC_SHAPE, n_filters=N_FILTERS,
                filter_width=FILTER_WIDTH, k_layers=K_LAYERS, n_blocks=N_BLOCKS, 
                context_layers=CONTEXT_LAYERS, context_blocks=CONTEXT_BLOCKS):
    
    # Skip connections
    skips = []
    
    # Dilation rates
    dilations_per_block = [2 ** i for i in range(k_layers)] * n_blocks
    dilations_per_ctx_block = [2 ** i for i in range(context_layers)] * context_blocks
    
    # Local Conditioning
    Input_lc = tf.keras.layers.Input(shape=lc_shape, dtype='float32', name='lc')
    lc = tf.keras.layers.Conv1D(n_filters, filter_width, padding='causal')(Input_lc)
    
    # Context stack
    for dilation_rate in dilations_per_ctx_block:
        
        # Convolución y gated activation
        lc_conv = tf.keras.layers.Conv1D(n_filters, filter_width, padding='causal',
                                        dilation_rate=dilation_rate)(lc)
        gated_act = tf.keras.layers.Multiply()([tf.keras.layers.Activation('tanh')(lc),
                                        tf.keras.layers.Activation('sigmoid')(lc)])
        gated_act = tf.keras.layers.Conv1D(1, 1, padding='same')(gated_act)
        
        # Residual connection
        lc = tf.keras.layers.Add()([lc, gated_act])
    
    # Input
    Input_seq = tf.keras.layers.Input(shape=inp_shape, dtype='float32', name='seq')
    seq = tf.keras.layers.Conv1D(n_filters, filter_width, padding='causal')(Input_seq) 

    for dilation_rate in dilations_per_block:

        # Sumamos la información del local conditioning a la convolución de los valores de audio
        seq_conv = tf.keras.layers.Conv1D(n_filters, filter_width, padding='causal',
                                        dilation_rate=dilation_rate)(seq)
        tmp = tf.keras.layers.Add()([seq_conv, lc])
        
        # Gated activation
        gated_act = tf.keras.layers.Multiply()([tf.keras.layers.Activation('tanh')(tmp),
                                        tf.keras.layers.Activation('sigmoid')(tmp)])

        gated_act = tf.keras.layers.Conv1D(1, 1, padding='same')(gated_act)

        # Residual connection
        seq = tf.keras.layers.Add()([seq, gated_act])    

        # Vamos añadiendo las 'Skip conections'
        skips.append(gated_act)

    # Sumamos todas las skip connections y computamos el resto de funciones
    out = tf.keras.layers.Add()(skips)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(1, 1, padding='same')(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(out_shape[1], 1, padding='same')(out)
    out = tf.keras.layers.Softmax()(out)
    return tf.keras.models.Model(inputs=[Input_seq, Input_lc], outputs=out)

model = get_wavenet()
model.compile(tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# ### Obtenemos los datos y los transformamos los datos a los que acepta el modelo

# In[31]:


# Parametros de preprocesado
MULTITHREAD = False

# Parámetros de entrenamiento
N_EPOCHS = 5
BATCH_SIZE = 160


# In[32]:


feature_description = {
    "lc": tf.io.FixedLenFeature(shape=[*LC_SHAPE], dtype=tf.float32),
    "seq": tf.io.FixedLenFeature(shape=[*INP_SHAPE], dtype=tf.int64),
    "out": tf.io.FixedLenFeature(shape=[*OUT_SHAPE], dtype=tf.int64)}

def parse_sample(sample):
    return tf.io.parse_single_example(sample, features=feature_description)

def midi_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))

def wav_feature(value, mulaw_enc):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def generate_data(df):
    
    # Para procesar los ficheros
    midi_hdlr = Midi_handler(sampling_rate=SAMPLING_RATE, dir_path=PATH, n_notes=N_NOTES)
    wav_hdlr = Wav_handler(sampling_rate=SAMPLING_RATE, dir_path=PATH)

    # Para aplicar el mu-law encoding
    mulaw_enc = Mu_law_encoder(mu=MU, n_bits=N_BITS, amps=AMPS)

    # Datos del local conditioning
    midi_data = midi_hdlr.vectorize_midi(df['midi_filename'])
    
    # Datos de la serie de tiempo
    wav_data = mulaw_enc.encode_series(wav_hdlr.vectorize_wav(df['audio_filename']))
    seq_data = tf.keras.utils.timeseries_dataset_from_array(wav_data[0:-1], None, sequence_length=INPUT_LEN, batch_size=None)
    out_data = tf.keras.utils.timeseries_dataset_from_array(wav_data[1:], None, sequence_length=INPUT_LEN, batch_size=None)

    for idx, (midi_, seq_, out_) in tqdm(enumerate(zip(midi_data, seq_data, out_data))):
        feature = {"lc": midi_feature(midi_), "seq": wav_feature(seq_, mulaw_enc), "out": wav_feature(out_, mulaw_enc)}

        yield tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def write_data(df, max_files=None, max_samples=None, file='out'):

    assert(file is not None)
    assert(max_samples is None or max_samples > INPUT_LEN)
    assert(max_files is None or max_files > 0)

    # Calculamos el número máximo de ficheros y ejemplos por fichero
    num_files = -1 if max_files is None else max_files
    num_samples = -1 if max_samples is None else max_samples
    
    serialized_features_dataset = tf.data.Dataset.from_generator(lambda : generate_data(df),
                    output_types=tf.string, output_shapes=())
        
    tf.data.experimental.save(serialized_features_dataset, file, compression='GZIP', shard_func=None)

def read_data(filename):
    assert(filename is not None)
    return  tf.data.experimental.load(filename, compression='GZIP')


# #### Empleamos multithreading para cachear los datos en ficheros

# In[19]:


if MULTITHREAD:

    class myThread(threading.Thread):

        def __init__(self, threadID, name, q):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.q = q

        def run(self):
            print(f"Levantando {self.name}...")
            process_data(self.name, self.q)
            print(f"... terminando {self.name}")

    def process_data(threadName, q):
        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                data = q.get()
                queueLock.release()
                print(f"{threadName} processing {data}\n")
                idx, split, midi, wav = data
                cache_file = 'cache/' + split + '/' + midi[:-5]
                write_data(df.iloc[idx], file=cache_file)
                dfLock.acquire()
                print(f"{threadName} actualiza dataframe\n")
                df.loc[idx, 'data_cached'] = True
                set_dataframe(df, 'maestro-v3.0.0.csv')
                dfLock.release()
            else:
                queueLock.release()
                time.sleep(1)

    N_THREADS = 3

    exitFlag = False
    queueLock = threading.Lock()
    dfLock = threading.Lock()
    workQueue = queue.Queue(len(df))

    threads = []

    # Create new threads
    for i in range(N_THREADS):
        thread = myThread(i, f"Thread {i}", workQueue)
        thread.start()
        threads.append(thread)

    # Fill the queue
    queueLock.acquire()
    for idx, (split, midi, wav, cached) in enumerate(zip(df['split'], df['midi_filename'], 
                                                     df['audio_filename'], df['data_cached'])):
        if not cached:
            workQueue.put((idx, split, midi, wav))
    queueLock.release()

    # Wait for queue to empty
    while not workQueue.empty():
        time.sleep(10)

    # Notify threads it's time to exit
    exitFlag = True

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting Main Thread")


# #### Para ejecutarlo en single-thread:

# In[ ]:


if not MULTITHREAD:

    # Creamos un fichero con los datos cacheados para cada uno de los archivos del dataset
    for idx, (split, midi, wav, cached) in enumerate(zip(df['split'], df['midi_filename'], 
                                                         df['audio_filename'], df['data_cached'])):
        if not cached:
            cache_file = 'cache/' + split + '/' + midi[:-5]
            write_data(df.iloc[idx], file=cache_file)
            df.loc[idx, 'data_cached'] = True
            set_dataframe(df, 'maestro-v3.0.0.csv')


# ### Preparamos el entrenamiento

# In[ ]:


training_progress_df = get_dataframe('traning_progress.csv')
training_progress_df['data_cached'] = df['data_cached']
training_progress_df.head()


# ### Loop de entrenamiento

# In[ ]:


try:
    model = tf.keras.models.load_model('model.hdf5')
    print("Cargado modelo anterior")
except:
    model = get_wavenet()
    model.compile(tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Creando nuevo modelo")

# Obtenemos el epoch actual
current_epoch = min(training_progress_df['epochs_trained'])

print(f'Comenzando el entrenamiento del modelo (nº de epochs: {N_EPOCHS}, parámetros a entrenar: {np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])})...')
print('------------------------------------------------------------------------')

while current_epoch < N_EPOCHS:

    # Obtenemos los ficheros a entrenar en este epoch
    train_df = training_progress_df[(training_progress_df['epochs_trained'] <= current_epoch) & 
                                    (training_progress_df['split'] == 'train') &
                                    (training_progress_df['data_cached'] == True)]
    
    print(f'Comenzando epoch nº {current_epoch}...')
    
    # Para cada fichero entrenamos
    for idx, (split, midi) in enumerate(zip(train_df['split'], train_df['midi_filename'])):
        
        print(f'Entrenando el fichero {midi[:-5]}...')
        
        # Leemos los datos y los preparamos
        train_dataset = read_data(f'cache/{split}/{midi[:-5]}')
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.map(parse_sample)
        train_dataset = train_dataset.map((lambda i: ({"lc": tf.expand_dims(i["lc"], 0), 
                                              "seq":  tf.one_hot(i["seq"], depth=N_LEVELS)}, 
                                             tf.one_hot(i["out"], depth=N_LEVELS))),
                                  num_parallel_calls = tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Calculamos la longitud
        length = int(sum(1 for _ in train_dataset) / BATCH_SIZE) - 1
        
        # Callbacks
        train_acc_logger = tf.keras.callbacks.CSVLogger('training_accuracy.csv')
        best_checkpoint = tf.keras.callbacks.ModelCheckpoint("best.hdf5", monitor='accuracy', 
                                                             save_best_only=True, mode='max')
        
        # Entrenamos
        history = model.fit(train_dataset, steps_per_epoch=length, epochs=1,                            callbacks=[train_acc_logger, best_checkpoint])
        
        # Guardamos el modelo
        model.save('model.hdf5')
        
        # Actualizamos todas las variables y actualizamos el csv con el progreso del entrenamiento
        training_progress_df.loc[df['midi_filename'] == midi, 'epochs_trained'] = current_epoch + 1
        set_dataframe(training_progress_df, 'traning_progress.csv')
        
    # Obtenemos los valores de cross validation
    crossval_df = training_progress_df[(training_progress_df['epochs_trained'] <= current_epoch) & 
                                    (training_progress_df['split'] == 'validation') &
                                    (training_progress_df['data_cached'] == True)]
    
    crossval_scores = []

    print('Calculando pérdida para el conjunto de cross validation...')
    
    # Para cada fichero medimos la cross validation score al final del epoch
    for idx, (split, midi) in enumerate(zip(crossval_df['split'], crossval_df['midi_filename'])):
        
        print(f'Calculando pérdida para el fichero {midi[:-5]}...')
        
        crossval_dataset = read_data(f'cache/{split}/{midi[:-5]}')
        crossval_dataset = crossval_dataset.cache()
        crossval_dataset = crossval_dataset.map(parse_sample)
        crossval_dataset = crossval_dataset.map((lambda i: ({"lc": tf.expand_dims(i["lc"], 0), 
                                              "seq":  tf.one_hot(i["seq"], depth=N_LEVELS)}, 
                                             tf.one_hot(i["out"], depth=N_LEVELS))),
                                  num_parallel_calls = tf.data.experimental.AUTOTUNE)
        crossval_dataset = crossval_dataset.batch(BATCH_SIZE)
        crossval_dataset = crossval_dataset.prefetch(tf.data.AUTOTUNE)

        # Calculamos la pedida
        statistics = model.evaluate(crossval_dataset)
        crossval_scores.append(statistics)
        
    with open("crossval_accuracy.csv", "a") as file:
        file.write(f"{np.mean([i[0] for i in crossval_scores])},{np.mean([i[1] for i in crossval_scores])}\n")

    break
    
    current_epoch += 1
    
    print('------------------------------------------------------------------------')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Estadísticas del entrenamiento

# In[ ]:


'''
    Helper function to plot keras history.
    https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
'''
def plot_history(history):
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Cross Validation'], loc='upper left')
    plt.show()
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Cross Validation'], loc='upper left')
    plt.show()


# In[ ]:


for h in histories:
    plot_history(h)


# ### Predicción

# In[ ]:


predictor = tf.keras.models.load_model('./best.hdf5')
predictor.summary()


# In[ ]:


'''
    Foo
'''
def predict_wav(in_file, out_file, dir_path, predictor, sr=SAMPLING_RATE, inp_shape=INP_SHAPE, enc=mulaw_enc):
    
    # TO DO: Quitar comment
    #pianoroll = vectorize_midis([dir_path + in_file], sampling_rate=sr, dir_path=dir_path)[0]
    
    # Listas empleadas para el input/output del preductor
    in_arr = tf.random.uniform(inp_shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
    lc_arr = []
    
    # Lista empleada para el output al fichero .wav
    wav_arr = tf.zeros([inp_shape[0]])
    
    # TO DO: Quitar [:]
    for i in tqdm(range(len(pianoroll[0:20000]))):
        
        # Getting the local conditioninng value for this step
        lc_arr = pianoroll[i]
        
        # Getting the value of the prediction
        prediction = predictor.predict({"seq": tf.expand_dims(in_arr, 0),                                         "lc": tf.expand_dims(tf.expand_dims(lc_arr, 0), 0)})[-1]
        prediction_idx = np.argmax(prediction[-1])
        
        # Figuring out the params for prediction in next step
        in_arr = tf.concat([in_arr[1:], tf.expand_dims(tf.keras.utils.to_categorical(prediction_idx, num_classes=N_LEVELS), 0)], 0)
        
        # Storing predicted values in wav format
        wav_arr = tf.concat([wav_arr, [enc.decode_sample(prediction_idx)]], 0)
        
    # Writting predicted series to file
    sf.write(dir_path + out_file, wav_arr, sr, 'PCM_24')
    
    return wav_arr


# In[ ]:


midi_hdlr = Midi_handler(sampling_rate=SAMPLING_RATE, dir_path=PATH, n_notes=N_NOTES)
in_file = train_df['midi_filename'][0]
out_file = 'prediction.wav'
dir_path = './i\:o'


# In[ ]:


wav_arr = predict_wav(in_file, out_file, dir_path, model)


# In[ ]:


pianoroll = (midi_hdlr.vectorize_midi(in_file) - MIDI_MIN) / (MIDI_MAX - MIDI_MIN)


# In[ ]:


print(pianoroll[0])


# In[ ]:





# In[35]:


wav_data = mulaw_enc.encode_series(wav_hdlr.vectorize_wav(df['audio_filename'][0]))


# In[87]:


in_data = tf.keras.utils.timeseries_dataset_from_array(wav_data[0:-1], None, sequence_length=INPUT_LEN, batch_size=None)
out_data = tf.keras.utils.timeseries_dataset_from_array(wav_data[1:], None, sequence_length=INPUT_LEN, batch_size=None)


# In[92]:


for i, j in zip(in_data, out_data):
    print(i.numpy()[100:150])
    print(j.numpy()[100:150])
    break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




