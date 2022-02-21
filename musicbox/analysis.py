import music21
import pandas as pd
import numpy as np

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

    chord_notes = _find_simultaneous_notes(144, df, True, True)
    str_note = _notes_to_string(chord_notes)

    # Three Methods to check out here:
    # 1. Take the bottom four tracks and find the chords for each fo these notes
    # 2. Find the lowest sounding note at every position of the disc and calculate the chords for that
    # 3. Sort everything by start time and then find a way to plow through everything by that




    import pdb; pdb.set_trace()


def _find_simultaneous_notes(note_id, data_array, include_previous = False, include_wider = False, as_string = False):
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

def _notes_to_string(notes_df, as_midi = False):
    if as_midi:
        chord_tones = notes_df["midi_note"].tolist()
        chord_tones = list(dict.fromkeys(chord_tones))
        chord_tones.sort()
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
    chord_tones = list(dict.fromkeys(chord_tones))
    chord_tones.sort()

    return "-".join(midi_to_note[note] for note in chord_tones)
