import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tqdm import tqdm

# Add your MIDI file paths here
midi_files = glob.glob("midi_files/*.mid")


def parse_midi_files(midi_files):
    notes = []
    instruments = set()
    for file in tqdm(midi_files, desc="Parsing MIDI files"):
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                for part in parts.parts:
                    instr = part.getInstrument()
                    instruments.add(instr.instrumentName)
                    notes_to_parse = part.recurse()
                    for element in notes_to_parse:
                        if isinstance(element, note.Note):
                            notes.append(
                                (str(element.pitch), instr.instrumentName))
                        elif isinstance(element, chord.Chord):
                            notes.append(
                                ('.'.join(str(n) for n in element.normalOrder), instr.instrumentName))
            else:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append((str(element.pitch), 'Piano'))
                    elif isinstance(element, chord.Chord):
                        notes.append(('.'.join(str(n)
                                     for n in element.normalOrder), 'Piano'))

            print(f"Successfully parsed {file}")
        except Exception as e:
            print(f"Error parsing {file}: {e}")
    return notes, instruments


# Parse the MIDI files
notes, instruments = parse_midi_files(midi_files)

# Save the notes for future use in a pickle file
notes_file_path = 'data/notes.pkl'
os.makedirs(os.path.dirname(notes_file_path), exist_ok=True)
with open(notes_file_path, 'wb') as f:
    pickle.dump(notes, f)

# Prepare sequences for the Transformer
sequence_length = 100
pitchnames = sorted(set(note for note, _ in notes))
instrumentnames = sorted(set(instr for _, instr in notes))
n_vocab = len(pitchnames)
n_instruments = len(instrumentnames)
note_to_int = {n: i for i, n in enumerate(pitchnames)}
instrument_to_int = {instr: i for i, instr in enumerate(instrumentnames)}

network_input = []
network_output = []
instrument_input = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[note] for note, _ in seq_in])
    instrument_input.append([instrument_to_int[instr] for _, instr in seq_in])
    network_output.append(note_to_int[seq_out[0]])

network_input = np.reshape(
    network_input, (len(network_input), sequence_length, 1))
instrument_input = np.reshape(
    instrument_input, (len(instrument_input), sequence_length, 1))
network_output = np.array(network_output)

# Save the prepared data
prepared_data = {
    'network_input': network_input,
    'instrument_input': instrument_input,
    'network_output': network_output,
    'note_to_int': note_to_int,
    'int_to_note': {i: n for n, i in note_to_int.items()},
    'instrument_to_int': instrument_to_int,
    'int_to_instrument': {i: instr for instr, i in instrument_to_int.items()},
    'n_vocab': n_vocab,
    'n_instruments': n_instruments,
    'sequence_length': sequence_length,
    'pitchnames': pitchnames,
    'instrumentnames': instrumentnames
}

prepared_data_file_path = 'data/prepared_data.pkl'
with open(prepared_data_file_path, 'wb') as f:
    pickle.dump(prepared_data, f)
