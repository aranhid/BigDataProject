import glob
from music21 import converter, instrument, note, chord, stream, duration as dur
import numpy
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LSTM, Activation, Reshape
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

FOLDER = 'midi'
sequence_length = 100
FILES_NUM = 5
EPOCH = 10
MODEL_FILE_NAME = FOLDER + '_Model_Seq' + str(sequence_length) + '_Files' + str(FILES_NUM) + '_Epoch' + str(EPOCH)

PATTERN_FILE = 'beet27m3'

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


oneTrackNotes = []
oneTrackDurations = []
for file in glob.glob(FOLDER +"/" + PATTERN_FILE + ".mid"):
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
            oneTrackNotes.append(str(element.pitch))
            oneTrackDurations.append(str(element.duration.type))
        elif isinstance(element, chord.Chord):
            oneTrackNotes.append('.'.join(str(n) for n in element.normalOrder))
            oneTrackDurations.append(element.duration.type)

# Получаем названия всех нот
pitchnames = sorted(set(item for item in notes))
durationnames = sorted(set(item for item in durations))
# Создаём словарь, чтобы привести названия нот к целым числам
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
duration_to_int = dict((duration, number) for number, duration in enumerate(durationnames))
network_input = []
network_output = []
# Создаём входные последовательности нот и соответствующие выходы
for i in range(0, len(oneTrackNotes) - sequence_length + 1, 1):
    sequence_in = list(zip(oneTrackNotes[i:i + sequence_length], oneTrackDurations[i:i+sequence_length]))
    sequence_out = list(zip(oneTrackNotes[i + sequence_length: i + 2 * sequence_length], oneTrackDurations[i + sequence_length: i + 2 * sequence_length]))
    if len(sequence_in) == len(sequence_out):
        network_input.append([(note_to_int[char], duration_to_int[duration]) for char, duration in sequence_in])
        network_output.append([(note_to_int[char], duration_to_int[duration]) for char, duration in sequence_out])
n_patterns = len(network_input)
print(n_patterns)
# Преобразовываем вводные данные в формат, совместимый со слоями LSTM
network_output = np.array(network_output)
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 2))
# Нормализуем входные данные
# n_vocab - кол-во нот
n_vocab = max(network_input[:, :, 0].max(), network_output[:, :, 0].max()) + 1 + 1
network_input = to_categorical(network_input, num_classes=188)
#network_output = to_categorical(network_output)

model = load_model('models/' + MODEL_FILE_NAME + '.h5')

start = numpy.random.randint(0, len(network_input)-1)
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
int_to_duration = dict((number, duration) for number, duration in enumerate(durationnames))
pattern = network_input[start]
prediction_output = []

for pat in pattern:
        index = numpy.argmax(pat[0])
        duration_i = numpy.argmax(pat[1])
        result = int_to_note[index]
        duration = int_to_duration[duration_i]
        prediction_output.append((result, duration))

# генерируем ноты
for note_index in range(1):
    prediction_input = pattern.reshape(1, sequence_length, 2, n_vocab)
    prediction_input = prediction_input / float(n_vocab)
    predictions = model.predict(prediction_input, verbose=0)[0]
    for prediction in predictions:
        index = numpy.argmax(prediction[0])
        duration_i = numpy.argmax(prediction[1])
        result = int_to_note[index]
        duration = int_to_duration[duration_i]
        prediction_output.append((result, duration))
        result = np.zeros_like(prediction)
        result[0][index] = 1
        result[1][duration_i] = 1
        pattern = np.concatenate([pattern, [result]])
    pattern = pattern[len(pattern) - sequence_length:len(pattern)]

offset = 0
output_notes = []
# Создаём последовательности нот и аккордов на основе значений, сгенерированных моделью
for pattern, duration in prediction_output:
    # pattern - аккорд
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            if (duration == 'complex'):
                duration = 'half'
            new_note.duration = dur.Duration(type=duration)
            new_note.storedInstrument = instrument.Accordion()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # pattern - нота
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Accordion()
        output_notes.append(new_note)
    # Немного смещаем последовательности
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output/' + MODEL_FILE_NAME + '.mid')
