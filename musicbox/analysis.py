import music21

def detect_key(proc):
    score = music21.converter.parse(proc.midi_filename)
    key = score.analyze('key')
    proc.key = key
    print("INFO: Key detected is " + key.tonic.name + " " + key.mode)