from midiutil.MidiFile import MIDIFile
import numpy as np

def _convert_track_degree(data_array, tracks_to_notes, degrees_per_beat):
    start_time = (360 - data_array[:,2]) / degrees_per_beat
    duration = data_array[:,3] / degrees_per_beat

    pitch = data_array[:,0]
    keys = np.array(list(tracks_to_notes.keys()))
    values = np.array(list(tracks_to_notes.values()))

    sidx = keys.argsort()

    ks = keys[sidx]
    vs = values[sidx]

    pitch = vs[np.searchsorted(ks, pitch)]

    # pitch = np.vectorize(tracks_to_notes.get)(data_array[:,0])
    return (start_time, duration, pitch)

def create_midi(data_array, additional_arguments):
    degrees_per_beat = 360 / additional_arguments["bars"]
    start_time, duration, pitch = _convert_track_degree(data_array, additional_arguments["track_mappings"], degrees_per_beat)
    midi_obj = MIDIFile(numTracks=1,
                removeDuplicates=False,  # set True?
                deinterleave=True,  # default
                adjust_origin=False,
                # default - if true find earliest event in all tracts and shift events so that time is 0
                file_format=1,  # default - set tempo track separately
                ticks_per_quarternote=480,  # 120, 240, 384, 480, and 960 are common values
                eventtime_is_ticks=False  # default
                )

    midi_obj.addTempo(0, 0, additional_arguments["bpm"])

    #for track_id in tpm.tracks_to_note.keys():
    #    midi_obj.addTempo(track_id, time=0, tempo=tpm.tempo)

    channel = 0 # we do not have multiple instruments
    volume = 100
    for i, _ in enumerate(start_time):
        midi_obj.addNote(track = 0,
                         channel = channel,
                         pitch = pitch[i],
                         time = start_time[i],
                         duration = duration[i],
                         volume = volume)

    with open(additional_arguments["filename"], "wb") as output_file:
        midi_obj.writeFile(output_file)