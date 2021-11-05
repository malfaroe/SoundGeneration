#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Data preprocessing tutorial implementation
#Url: https://youtu.be/coEgwnMBuo0

import music21 as m21
import os
import json
import tensorflow.keras as keras
import numpy as np 


# In[17]:


"""Steps
1. Load data
2. Filter songs with no acceptable duration
3. Transpose songs yo cmaj/Amin
4. Encode songs with music time series representation
5. save songs to text file
6. Create s ingle file to store all songs
7. Mapping all the symbols of the songs with integers for the nn to read
8. Use the mapping to convert the single file of songs into integers
9. Create sequences,which are the way we feed the LSTM neural network engine"""


# In[18]:


#Path: C:\Users\malfaro\Desktop\mae_code\SoundGeneration


# In[19]:


#1. Data loading

###MAC PATHS 
#DATASET_PATH = r"/Users/mauricioalfaro/Documents/mae_code/SoundGeneration/data/essen/europa/deutschl/test/"
DATASET_PATH = r"/Users/mauricioalfaro/Documents/mae_code/SoundGeneration/data/essen/europa/deutschl/erk/"


SAVE_DIR = r"/Users/mauricioalfaro/Documents/mae_code/SoundGeneration/dataset" #directorio donde va a guardarse todo

###WINDOWS PATHS
#DATASET_PATH = r"C:\Users\malfaro\Desktop\mae_code\SoundGeneration\data\essen\europa\deutschl\test"
#SAVE_DIR = r"C:\Users\malfaro\Desktop\mae_code\SoundGeneration\dataset" #directorio donde va a guardarse todo




SINGLE_FILE_PATH = "single_dataset" #este es un archivo que se crea con ese nombre (text file)
#r"/Users/mauricioalfaro/Documents/mae_code/SoundGeneration/single_file_dataset"
#ahi arriba se guardaran finalmente todas las canciones en un solo archivo
#MAPPING_PATH = r"/Users/mauricioalfaro/Documents/mae_code/SoundGeneration/mapping.json"
MAPPING_PATH = "mapping.json" #archivo json que se creara con ese nombre


#Go through all the .kern files and load them together using m21
def load_songs_in_kern(dataset_path):
   songs = []
   for path, subdirs, files in os.walk(dataset_path):
        for file in files:
           if file[-3:] == "krn":
               song = m21.converter.parse(os.path.join(path, file))#convertir a objeto de music21
               songs.append(song)

   return songs

def preprocess(dataset_path):
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs!")
              
              
#2. Filter by acceptable duration

ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]

def has_acceptable_duration(song, acceptable_durations):
    """Boolean method for checking if the songs complies with duration.
    Se considera como referncia una negra (quarter length)
    redonda = whole note = 4
    blanca = half note = 2
    blanca con punto = 3
    negra = quarter note = 1
    negra con punto = 1.5
    corchea = eigth note = 0.5
    corchea con punto = 0.75
    semicorchea = sixteenth note = 0.25
    """
    for note in song.flat.notesAndRests: 
        #flat toma todos los objetos de la cancion, los convierte en lista
        #notesAndRests deja solo las notas y silencios, excluyendo claves, simolos, etc
        if note.duration.quarterLength not in acceptable_durations:
            return False
        
        return True
    
def transpose(song):
    """
    - Detect the key or estimate it using music21
    - get the interval or distance necessary to transpose to Cmaj/Amin
    - transpose using m21 if necessary"""
    #Get the song key
    #usually the key is in the first measure of the song
    parts = song.getElementsByClass(m21.stream.Part) # extracts the parts adnd extracts all the elements by part 
    #go to the first part and take all the measures in part 0 
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4] #tomo la primera parte de measures y extraigo de esa lista el elemento 4 que es key
    
    #In case the key is not in the song we use m21 to estimate it
    
    if not isinstance(key, m21.key.Key):#if the song doesnt have a key stored
        key = song.analyze("key") #estimate it...
    #Now transpose to cmaj or A minor depending on the mode of the song...
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C")) #calculates the interval
        
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A ")) #calculates the interval
    
    #print("Original key:", key)
    #transpose de song
    transposed_song = song.transpose(interval)
    
    return transposed_song


#4. Encode songs with music time series representation

def encode_song(song, time_step = 0.25):
    """takes a song as a music21 object
    and returns a string in which the song has
    been encoded into a time series music representation
    Example:
    a note of pitch 60 that lasts one bar would be encoded
    as: [60,"_", "_", "_"]
    Time_step = 0.25 significa que nos vamos moviendo en semicorcheas por
    toda la canción"""
    encoded_song = []


    for event in song.flat.notesAndRests:#flat crea una lista de todos los elementos de la cancion
        """un event es una nota o rest. Por ejemplo: la canción empieza con 
        una nota larga de pitch 60 que dura 4 tiempos (un compás)"""
        #pueden ser notes or rests
        #if note ---> guardar la nota
        if isinstance(event, m21.note.Note):#si el evento es una nota
            symbol = event.pitch.midi #guarda la nota como midi (60 en este caso)
        #if rest---> guardar como string "r"
        if isinstance(event, m21.note.Rest):
            symbol = "r"
    #ahora convierte todo a time series music notation. El evento del ejemplo
    #quedaria como [60,"_", "_", "_"] steps es en nro de timesteps que dura el evento. 
    #Para calcularlo tomo la duracion del evento en negras y la divido por time_step"""
        steps = int(event.duration.quarterLength / time_step)
        
        #tomo e evento dividido en steps y si estoy al comnienzo guardo el simbolo, si no 
        #guardo "_", ya que siempre va a ser así
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
            
    #cast the encoded song into a string
    #convierto con map todos los caracteres de encoded_song a str
    #y luego los uno separados por un " "
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

    
    
#6. create a single for the whole dataset

SEQUENCE_LENGTH = 64 #se usara para delimitar el inicio de una nueva cancion

def load(dataset_path):#metodo para leer las canciones de su directorio de origen
    with open(dataset_path, "r") as fp:
        song = fp.read() 
    return song
    
def create_single_file_dataset(dataset_path, single_file_path, sequence_length):
    
#se crea un gran string donde se almacenan todas las encoded songs separadas por un delimitador
    new_song_delimiter = "/ " * sequence_length 
    #slash y espacio repetidos 64 veces delimitando, esto porque asi los lee las rnn/lstm
    
    songs = "" #inicializo el string

    #load songs and add delimiters
    for path, _, files in os.walk(dataset_path): #paseo por todo el directorio de canciones individuales
        for file in files:
            file_path = os.path.join(path, file) #averiguo la ubicacion exacta de la cancion
            song = load(file_path) #metodo que load la cancion desde el directorio
            songs = songs + song + " " + new_song_delimiter
            
    songs = songs[:-1] #recorto el espacio que quedaria en el delimitador de la ultima cancion
    
    #save string that contains all dataset en su directorio
    #save_path = os.path.join(single_file_path, "single_dataset") 

    with open(single_file_path, "w") as fp:
        fp.write(songs)
    
    return songs 
    

#Create a mapping for the song symbols
def create_mapping(songs, mapping_path):
    mappings = {}
    #create the mapping
    songs = songs.split() #separa todos los elementos de songs en una lista
    vocabulary = list(set(songs)) #set toma los elementos unicos de song y list los convert into list
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
        
    #Save the mapping into a json file for using it later
    #save_path = os.path.join(mapping_path, "mapping_json") 
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent= 4)
 

#Convert the single file into integers using the mapping

def convert_songs_into_integers(songs):
    int_songs = [ ] #vaciaremos el mapeo a una lista
    
    #load the mappings file (json) that contains the dictionary
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
        
    #cast songs string into a list (recordar que songs es un string)
    songs = songs.split()
    
    #map songs into int
    for symbol in songs:
        int_songs.append(mappings[symbol])
       
    return int_songs


#Generating training sequences...
#las LSTM se estructuran tomando una secuencia de notas y prediciendo cual es la proxima
#Por ser supervisado, se le da una secuencia y se le muestra un target; asi se va entrenando
#Por ello tomaremos una secuencia de 64 time_steps (que equivalen a 4 compases de 4/4) como sample
#y como target le mostramos la siguiente nota o figura
#Para ello las secuencias se construyen considerando que se trata de un time series, mviendose
#con un window hacia adelante
#En este caso, dado que tenemos un sequence length de 64 timesteps, si hay 100 symbols en total
#y nos movemos de a uno en la ventana, tendriamos un total de secuencias de 100 - 64

def generate_training_sequences(sequence_length):
    
    #load the songs and map them to int
    songs = load(SINGLE_FILE_PATH)
    int_songs = convert_songs_into_integers(songs)
    
    #generate the training sequences
    inputs = [] #guardar cada secuencia
    targets = [] #guardar los targets asociados a cada secuencia
    num_sequences = len(int_songs) - sequence_length  #cantidad de secuencias generables
    
    for i in range(num_sequences):
        inputs.append(int_songs[i:sequence_length + i])
        targets.append(int_songs[sequence_length + i])
    #one-hot encoding
    vocabulary_size = len(set(int_songs)) #nro de symbolos unicos, que son las categorias a encode
    inputs = keras.utils.to_categorical(inputs, num_classes= vocabulary_size)
    
    #Convert the targets into a numpy array
    targets = np.array(targets)
    
    return inputs, targets
    

    
def preprocess(dataset_path):
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs!")
    #Filter by duration
    for i, song in enumerate(songs):
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue #si la cancion no cumple la ignora
        #Transpose song
        song = transpose(song)
        # Encode songs with music time series representation
        encoded_song = encode_song(song)
        
        #5. save songs to text file  
        save_path = os.path.join(SAVE_DIR, str(i)) #guarda cada cancion con un nro en el dir "dataset"
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

            

    
def main():
    preprocess(DATASET_PATH)    
    songs= create_single_file_dataset(SAVE_DIR,SINGLE_FILE_PATH, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)



if __name__ == "__main__":
    main()


# In[ ]:


#preprocess(DATASET_PATH)    
#songs= create_single_file_dataset(SAVE_DIR,SINGLE_FILE_PATH, SEQUENCE_LENGTH)
#create_mapping(songs, MAPPING_PATH)
#inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


