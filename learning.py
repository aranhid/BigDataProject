import glob
from music21 import converter, instrument, note, chord, stream
import numpy
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Activation, Reshape

FOLDER = 'midi'
FILES_NUM = 5
sequence_length = 100
EPOCH = 10
MODEL_FILE_NAME = FOLDER + '_Model_Seq' + str(sequence_length) + '_Files' + str(FILES_NUM) + '_Epoch' + str(EPOCH)

notes = []
durations = []
for file in glob.glob(FOLDER +"/*.mid")[:FILES_NUM]:
    print(file)
    midi = converter.parse(file)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
            durations.append(str(element.duration.type))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
            durations.append(element.duration.type)

# Получаем названия всех нот
pitchnames = sorted(set(item for item in notes))
durationnames = sorted(set(item for item in durations))
# Создаём словарь, чтобы привести названия нот к целым числам
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
duration_to_int = dict((duration, number) for number, duration in enumerate(durationnames))
network_input = []
network_output = []
# Создаём входные последовательности нот и соответствующие выходы
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = list(zip(notes[i:i + sequence_length], durations[i:i+sequence_length]))
    sequence_out = list(zip(notes[i + sequence_length: i + 2 * sequence_length], 
                            durations[i + sequence_length: i + 2 * sequence_length]))
    if len(sequence_in) == len(sequence_out):
        network_input.append([(note_to_int[char], duration_to_int[duration]) for char, duration in sequence_in])
        network_output.append([(note_to_int[char], duration_to_int[duration]) for char, duration in sequence_out])
n_patterns = len(network_input)
# Преобразовываем вводные данные в формат, совместимый со слоями LSTM
network_output = np.array(network_output)
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 2))
# Нормализуем входные данные
# n_vocab - кол-во нот
n_vocab = max(network_input[:, :, 0].max(), network_output[:, :, 0].max()) + 1
network_input = to_categorical(network_input)
network_output = to_categorical(network_output)

model = Sequential()
model.add(Reshape((network_input.shape[1] * network_input.shape[2], network_input.shape[3])))
model.add(LSTM(256,
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(2*n_vocab * sequence_length))
model.add(Reshape((sequence_length, 2, n_vocab)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(network_input, network_output, epochs=EPOCH, callbacks=[ModelCheckpoint('models/' + MODEL_FILE_NAME + '.h5')])