import music21
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from musicbox.helpers import make_color_image, midi_to_notes
from datetime import datetime


def detect_key(proc):
    """Detects the key of the generated midi file using the music21 librarys key detection methods

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method
    """    
    score = music21.converter.parse(proc.midi_filename)
    key = score.analyze('key')
    proc.key = key
    write_to = open("keys.txt", "a")
    write_to.write(f"{key.tonic.name} {key.mode}\n")
    write_to.close()
    print("INFO: Key detected is " + key.tonic.name + " " + key.mode)

def find_harmonies_bass(proc):
    """Finds chords in the processed medium by finding notes sounding concurrently with bass notes

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method
    """

    data = pd.DataFrame(data = proc.data_array, columns = ["note_id", "start_time", "duration", "midi_note"])
    data = data.astype({"note_id": int, "midi_note": int})
    data = data.set_index("note_id")
    data["end_time"] = data["start_time"] + data["duration"]
    data["chord_id"] = pd.Series(dtype = "int")

    bass_notes = data[data["midi_note"] <= proc.parameters["bass_cutoff"]]

    mapping = {}
    unique_chords = []
    chords = []

    for idx, row in bass_notes.iterrows():
        chord_notes = _find_simultaneous_notes(idx, data[data["chord_id"].isna()], include_previous=False)
        chord_str = _format_chord(chord_notes, True)
        # chord_str = _notes_to_string(chord_notes, False, False)
        chords.append(chord_str)

        mapping[idx] = chord_notes.index.to_list()


        # This will deduplicate chord ids in the data array and use ascending ids

        if not chord_str in unique_chords:
            unique_chords.append(chord_str)

        chord_id = unique_chords.index(chord_str)
        data.loc[chord_notes.index, "chord_id"] = chord_id

    base_filename = "chords_bass_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    _make_chord_image(data, proc, unique_chords, base_filename)

    _chords_to_file(unique_chords, base_filename)

def find_harmonies_seq(proc):
    """Finds chords in the processed medium by sequentially walking through all notes and finding the ones that sound concurrently

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method
    """

    data = pd.DataFrame(data = proc.data_array, columns = ["note_id", "start_time", "duration", "midi_note"])
    data = data.astype({"note_id": int, "midi_note": int})
    data = data.set_index("note_id")
    data["end_time"] = data["start_time"] + data["duration"]
    data["chord_id"] = pd.Series(dtype = "int")
    data.sort_values(by = ["start_time"])

    mapping = {}
    unique_chords = []
    chords = []
    chord_id = 1


    start_time = data[data["chord_id"].isna()]["start_time"].min()

    while start_time < 360:
        print("Starting at", start_time)

        chunk = data[(data["start_time"].between(start_time, start_time + proc.parameters["lookahead"])) & (data["chord_id"].isna())]

        print(len(chunk), "Notes in chunk")
        
        for idx, row in chunk[(chunk["midi_note"] == chunk["midi_note"].min()) & (chunk["midi_note"] <= proc.parameters["cutoff"])].iterrows():
            chord_notes = _find_simultaneous_notes(idx, data[data["chord_id"].isna()], include_previous=False)
            chord_str = _format_chord(chord_notes, True)
            # chord_str = _notes_to_string(chord_notes, False, False)
            chords.append(chord_str)

            mapping[idx] = chord_notes.index.to_list()


            # This will deduplicate chord ids in the data array and use ascending ids

            if not chord_str in unique_chords:
                unique_chords.append(chord_str)

            chord_id = unique_chords.index(chord_str)
            data.loc[chord_notes.index, "chord_id"] = chord_id
            # chunk.loc[chord_notes.index, "chord_id"] = chord_id

        start_time = start_time + proc.parameters["lookahead"]

    base_filename = "chords_bass_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    _make_chord_image(data, proc, unique_chords, base_filename)

    _chords_to_file(unique_chords, base_filename)

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
        conditions = conditions | (data_array["end_time"].between(start_time, end_time, inclusive = "both"))

    if include_wider:
        conditions = conditions | ((data_array["start_time"] <= start_time) & (data_array["end_time"] >= end_time))
    
    sim_notes = data_array[conditions]

    return sim_notes

def _format_chord(notes_df, as_midi = False):
    notes_df = notes_df.sort_values(by=["midi_note"])
    notes_df["base_note"] = notes_df["midi_note"].mod(12)
    chord_str = ""
    tone_map = midi_to_notes()

    while len(notes_df) > 0:
        row = notes_df.iloc[0]
        duplicate_notes = notes_df[notes_df["base_note"] == row["base_note"]]
        if as_midi:
            chord_str = chord_str + "-".join([str(note) for note in duplicate_notes["midi_note"].tolist()]) + "\n"
        else:
            chord_str = chord_str + "-".join([tone_map[note] for note in duplicate_notes["midi_note"].tolist()]) + "\n"
        notes_df = notes_df.drop(notes_df[notes_df["base_note"] == row["base_note"]].index)

    return chord_str

def _make_chord_image(data_array, proc, filename, chord_names = None):
    base_image = proc.labels.copy().astype(np.uint16)

    unassigned_color = data_array["chord_id"].max() + 1

    data_array[data_array["chord_id"].isna()] = unassigned_color

    for idx, row in data_array.iterrows():
        base_image[base_image == idx] = row.chord_id

    x_start = 100
    y_coord = 100
    for chord_id in data_array["chord_id"].unique():
        if chord_names is not None and chord_id in chord_names:
            text = chord_names[chord_id]

            for i, line in enumerate(text.split("\n")):
                y_actual = y_coord + i * 100
                cv2.putText(base_image, line, (x_start, y_actual), cv2.FONT_HERSHEY_COMPLEX, 2, chord_id, 2 )

            x_start = x_start + 500

    cv2.putText(base_image, "Unassigned", (proc.center_x, proc.center_y), cv2.FONT_HERSHEY_COMPLEX, 2, unassigned_color, 1)

    image = make_color_image(base_image)
    cv2.imwrite(filename + ".tiff", image)
    

def _chords_to_file(chords, filename):
    with open(filename + ".txt", "w") as f:
        f.writelines(f'{chord}\n' for chord in chords)

def plot_note_frequencies(proc):
    """Creates plots of the frequencies and lengths each notes appear in the processed medium

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method
    """

    plt.ioff()

    print("hi!")

    tone_map = midi_to_notes()
    data = pd.DataFrame(data = proc.data_array, columns = ["note_id", "start_time", "duration", "midi_note"])
    data = data.astype({"note_id": int, "midi_note": int})
    data = data.set_index("note_id")

    data_notes = data.replace({"midi_note": tone_map})
    
    data_notes_count = data.groupby(["midi_note"])["midi_note"].count()
    # We can do this or not, need to get feedback
    # We might also want to use only notes available to the specific disc?
    data_notes_count = data_notes_count.reindex(list(range(data_notes_count.index.min(), data_notes_count.index.max() + 1)), fill_value = 0)
    # Better or worse? Don't know, we'll see
    data_notes_count = data_notes_count.rename(index = tone_map)
    positions = np.arange(len(data_notes_count))
    figure = plt.figure(figsize = (16, 5))
    plt.bar(positions, data_notes_count, align = "center", alpha = 0.5)
    plt.xticks(positions, data_notes_count.index)
    plt.ylabel("Frequency")
    # plt.title("Notenhäufigkeit")
    plt.savefig(os.path.join(proc.parameters["debug_dir"], "notes_frequency.png"))
    plt.close()

    data_note_lengths = data.groupby(["midi_note"])["duration"].sum()
    # Same things as before
    data_note_lengths = data_note_lengths.reindex(list(range(data_note_lengths.index.min(), data_note_lengths.index.max() + 1)), fill_value = 0)
    data_note_lengths = data_note_lengths.rename(index = tone_map)
    positions = np.arange(len(data_note_lengths))
    figure = plt.figure(figsize = (16, 5))
    plt.bar(positions, data_note_lengths, align = "center", alpha = 0.5)
    plt.xticks(positions, data_note_lengths.index)
    plt.ylabel("Anzahl")
    plt.title("Notenlänge")
    plt.savefig(os.path.join(proc.parameters["debug_dir"], "notes_duration.png"))
    plt.close()

    # Now we do the same thing but we dont care about the octave

    midi_to_note = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "Bb",
        11: "B"
    }

    data_base_notes = data.copy()
    data_base_notes["midi_note"] = [tone % 12 for tone in data_base_notes["midi_note"].tolist()]

    base_note_count = data_base_notes.groupby(["midi_note"])["midi_note"].count()
    # We can do this or not, need to get feedback
    # We might also want to use only notes available to the specific disc?
    base_note_count = base_note_count.reindex(list(range(0, 12)), fill_value = 0)
    # Better or worse? Don't know, we'll see
    base_note_count = base_note_count.rename(index = midi_to_note)
    positions = np.arange(len(base_note_count))
    figure = plt.figure(figsize = (8, 5))
    plt.bar(positions, base_note_count, align = "center", alpha = 0.5)
    plt.xticks(positions, base_note_count.index)
    plt.ylabel("Anzahl")
    # plt.title("Notefrequency (ohne Oktaven)")
    plt.savefig(os.path.join(proc.parameters["debug_dir"], "base_note_frequencies.png"))
    plt.close()

    base_note_count = data_base_notes.groupby(["midi_note"])["duration"].sum()
    # We can do this or not, need to get feedback
    # We might also want to use only notes available to the specific disc?
    base_note_count = base_note_count.reindex(list(range(0, 12)), fill_value = 0)
    # Better or worse? Don't know, we'll see
    base_note_count = base_note_count.rename(index = midi_to_note)
    positions = np.arange(len(base_note_count))
    figure = plt.figure(figsize = (8, 5))
    plt.bar(positions, base_note_count, align = "center", alpha = 0.5)
    plt.xticks(positions, base_note_count.index)
    plt.ylabel("Anzahl")
    plt.title("Notenlänge (ohne Oktaven)")
    plt.savefig(os.path.join(proc.parameters["debug_dir"], "base_note_durations.png"))
    plt.close()

def find_last_chords(proc):
    """Finds the last sounding chords on the processed medium

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method
    """    

    n = 2

    data = pd.DataFrame(data = proc.data_array, columns = ["note_id", "start_time", "duration", "midi_note"])
    data = data.astype({"note_id": int, "midi_note": int})
    data = data.set_index("note_id")
    data["end_time"] = data["start_time"] + data["duration"]
    data["chord_id"] = pd.Series(dtype = "int")

    chord_names = dict()

    i = 0
    while i < n:
        note_id = data[data["chord_id"].isna()]["start_time"].idxmax()
        note_start = data.loc[[note_id]]["start_time"].iloc[0]
        
        chunk = data[(data["start_time"].between(note_start - 0.5, note_start + 0.5)) & (data["chord_id"].isna())]

        chord = _find_simultaneous_notes(note_id, data, True, True)
        # Or we might just want to do
        # chord = chunk
        
        if (chord.shape[0] > 2):
            i = i + 1
            data.loc[chunk.index, "chord_id"] = i
            chord_names[i] = _format_chord(chord, False)
            # Generate chord name here
        else:
            data.loc[chunk.index, "chord_id"] = -1

    data[data["chord_id"] == -1] = np.nan
    _make_chord_image(data, proc, os.path.join(proc.parameters["debug_dir"], "last_chords"), chord_names)

def make_streamplot(proc):
    """Creates a streamgraph diagram of the processed medium

    Args:
        proc (musicbox.Processor.processor): The processor instance that called the method
    """

    time = [0,30,60,90,120,150,180,210,240,270,300,330,360]
    labels = list(range(1, len(time)))
    data = pd.DataFrame(data = proc.data_array, columns = ["note_id", "start_time", "duration", "midi_note"])

    data["bin"] = pd.cut(data.start_time, bins = time, labels=labels)

    midi_to_note = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "Bb",
        11: "B"
    }

    data["midi_note"] = [midi_to_note[tone % 12] for tone in data["midi_note"].tolist()]

    data = data.groupby(["midi_note", "bin"])
    
    size = data.size()

    freq_by_note = {}

    for note in size.index.get_level_values(0).unique().tolist():
        freq_by_note[note] = size[note].tolist()


    plt.ioff()
    fig, ax = plt.subplots()
    ax.stackplot(time[:-1], freq_by_note.values(),
                labels=freq_by_note.keys(), alpha=0.8)
    ax.legend(loc='upper right')
    ax.set_xlabel('Disc Part')
    ax.set_ylabel('Note Frequency')
    plt.savefig(os.path.join(proc.parameters["debug_dir"], "streamplot.png"))
    plt.close()
