import flask
import os
import sys
import inspect

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return flask.redirect("/tool/index.html")

@app.route('/tool/<path:path>', methods = ['GET'])
def send_map(path):
    return flask.send_from_directory("../client", path)

def run_server():
    app.run(use_reloader = False)