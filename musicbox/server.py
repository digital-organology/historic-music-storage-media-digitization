import flask
import os
import sys
import inspect
import numpy as np
from musicbox.notes.convert import convert_notes
from musicbox.notes.midi import create_midi

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.shape_ids = []
app.shape_polygons = []
app.center_x = 0
app.center_y = 0

@app.route('/', methods=['GET'])
def home():
    return flask.redirect("/tool/index.html")

@app.route('/tool/<path:path>', methods = ['GET'])
def send_map(path):
    return flask.send_from_directory("../client", path)

@app.route('/generate-midi', methods = ['POST'])
def generateMidi():
    # import pdb; pdb.set_trace()
    # json = dict(zip(list(map(int, list(app.shape_dict.keys()))), list(map(int, list(app.assignments)))))
    # return json

    data = flask.request.json

    assignments = np.array(list(data.items())).astype(np.int64)
    assignments = assignments[assignments[:,0].argsort()]

    arr = np.column_stack((app.notes, assignments[:,1]))

    per = [4, 1, 2, 3, 0]
    arr[:] = arr[:,per]

    create_midi(arr, app.track_mappings, 144, "flask_output.mid", 200)

    return flask.send_file("../flask_output.mid")
    # shape_ids = []

    

def run_server(shape_dict, track_mappings, center_x, center_y):
    app.notes = convert_notes(shape_dict.values(), list(shape_dict.keys()), center_x, center_y)
    app.track_mappings = track_mappings
    app.run(use_reloader = False)