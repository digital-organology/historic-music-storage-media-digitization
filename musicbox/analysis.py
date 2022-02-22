import music21
import pandas as pd
import numpy as np
import cv2
import collections
from musicbox.helpers import gen_lut, make_color_image


def detect_key(proc):
    score = music21.converter.parse(proc.midi_filename)
    key = score.analyze('key')
    proc.key = key
    print("INFO: Key detected is " + key.tonic.name + " " + key.mode)

def find_harmonies(proc):
    df = pd.DataFrame(data = proc.data_array, columns = ["note_id", "start_time", "duration", "midi_note"])
    df = df.astype({"note_id": int, "midi_note": int})
    df = df.set_index("note_id")
    df["end_time"] = df["start_time"] + df["duration"]

    # Three Methods to check out here:
    # 1. Take the bottom four tracks and find the chords for each fo these notes
    # 2. Find the lowest sounding note at every position of the disc and calculate the chords for that
    # 3. Sort everything by start time and then find a way to plow through everything by that

    data = df.copy()
    data["chord_id"] = pd.Series(dtype = "int")
    bass_notes = data[data["midi_note"] <= 52]

    mapping = {}
    unique_chords = []
    chords = []

    for idx, row in bass_notes.iterrows():
        chord_notes = _find_simultaneous_notes(idx, data[data["chord_id"].isna()])
        chord_str = _notes_to_string(chord_notes)
        chords.append(chord_str)

        mapping[idx] = chord_notes.index.to_list()


        # This will deduplicate chord ids in the data array and use ascending ids

        if not chord_str in unique_chords:
            unique_chords.append(chord_str)

        chord_id = unique_chords.index(chord_str)
        data.loc[chord_notes.index, "chord_id"] = chord_id

        # This would use unique ids for each occurence
        # data.loc[chord_notes.index, "chord_id"] = idx

    chord_freq = collections.Counter(chords)

    _make_chord_image(data, proc)
    print(chord_freq)





def _find_simultaneous_notes(note_id, data_array, include_previous = False, include_wider = False):
    """Find notes that sound simultaneous with the specified note

    Args:
        note_id (int): Note for which to find simultaneous notes
        data_array (DataFrame): Data in which to search for the notes
        include_previous (bool, optional): Whather to include notes that started before the specified note but still sound. Defaults to False.
        include_wider (bool, optional): Whather to include notes that started before the specified note and end after it. Defaults to False.
    """    
    note = data_array.T[note_id]
    start_time = note.start_time
    end_time = note.end_time

    conditions = (data_array["start_time"].between(start_time, end_time, inclusive = "both"))

    if include_previous:
        conditions = conditions | (data_array["end_time"].between(end_time, start_time, inclusive = "both"))

    if include_wider:
        conditions = conditions | ((data_array["start_time"] >= start_time) & (data_array["end_time"] <= end_time))
    
    sim_notes = data_array[conditions]

    return sim_notes

def _notes_to_string(notes_df, as_midi = False, deduplicate = True):
    notes_df = notes_df.sort_values(by=["midi_note"])

    if as_midi:
        chord_tones = notes_df["midi_note"].tolist()
        if deduplicate:
            chord_tones = list(dict.fromkeys(chord_tones))
        return "-".join(str(note) for note in chord_tones)

    midi_to_note = {
        21: "A",
        22: "Bb",
        23: "B",
        24: "C",
        25: "C#",
        26: "D",
        27: "D#",
        28: "E",
        29: "F",
        30: "F#",
        31: "G",
        32: "G#"
    }

    chord_tones = [tone % 12 + 21 for tone in notes_df["midi_note"].tolist()]
    if deduplicate:
        chord_tones = list(dict.fromkeys(chord_tones))

    return "-".join(midi_to_note[note] for note in chord_tones)


def _make_chord_image(data_array, proc):
    base_image = proc.labels.copy()

    data_array[data_array["chord_id"].isna()] = data_array["chord_id"].max() + 1

    for idx, row in data_array.iterrows():
        base_image[base_image == idx] = row.chord_id

    image = make_color_image(base_image)
    cv2.imwrite("chords.png", image)
    