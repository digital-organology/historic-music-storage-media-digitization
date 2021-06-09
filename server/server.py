import flask
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import musicbox

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return flask.redirect("/tool/index.html")

@app.route('/tool/<path:path>', methods = ['GET'])
def send_map(path):
    return flask.send_from_directory("../client/", path)

app.run()