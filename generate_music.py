import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from music21 import stream, note, chord, instrument

# Load the prepared data
with open('data/prepared_data.pkl', 'rb') as f:
    prepared_data = pickle.load(f)

network_input = prepared_data['network_input']
instrument_input = prepared_data['instrument_input']
int_to_note = prepared_data['int_to_note']
int_to_instrument = prepared_data['int_to_instrument']
n_vocab = prepared_data['n_vocab']
sequence_length = prepared_data['sequence_length']

# Load the saved model
model = models.load_model('saved_model/best_model')
print("Model loaded from 'saved_model/best_model'")

# Function to generate music


def generate_music(model, int_to_note, int_to_instrument, sequence_length, n_vocab, output_file='output.mid'):
    # Choose a random seed sequence from training data.
    start_index = np.random.randint(0, len(network_input) - 1)
    input_seq = network_input[start_index]
    instrument_seq = instrument_input[start_index]

    prediction_output = []
    pattern = list(input_seq)
    instrument_pattern = list(instrument_seq)

    # Generate notes.
    for note_index in range(500):  # Generate 500 notes
        input_seq = np.reshape(pattern, (1, len(pattern)))
        instrument_seq = np.reshape(
            instrument_pattern, (1, len(instrument_pattern)))
        prediction = model.predict([input_seq, instrument_seq], verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(
            (result, int_to_instrument[instrument_pattern[-1]]))
        pattern.append(index)
        instrument_pattern.append(instrument_pattern[-1])
        pattern = pattern[1:]
        instrument_pattern = instrument_pattern[1:]

    # Convert prediction_output into a MIDI stream.
    offset = 0
    output_notes = []
    for pattern, instr in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes_in_chord = [int(n) for n in notes_in_chord]
            new_chord = chord.Chord(notes_in_chord)
            new_chord.offset = offset
            new_chord.storedInstrument = instrument.fromString(instr)
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.fromString(instr)
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"Generated music saved to {output_file}")


# Generate music
generate_music(model, int_to_note, int_to_instrument, sequence_length, n_vocab)
