from spheres import *

sphere = Sphere(state=qt.rand_ket(2),\
                energy=qt.rand_herm(2),\
                dt=0.01,\
                evolving=True,\
                show_phase=True,\
                show_components=False,\
                show_husimi=False,\
                show_projection=False)

##################################################################################################################

import os
import json
import time
import gevent
import logging
from flask import Flask, request, Response, render_template
import socketio

sio = socketio.Server()
app = Flask("spheres")
app.wsgi_app = socketio.Middleware(sio, app.wsgi_app)
thread = None

#import logging
#log = logging.getLogger('werkzeug')
#log.disabled = True
#app.logger.disabled = True

def animate():
    global sphere
    while True:
        sphere.update()
        phase = sphere.phase() if sphere.show_phase else []
        stuff = sphere.allstars(plane_stars=sphere.show_projection,\
                                component_stars=sphere.show_components,\
                                plane_component_stars=sphere.show_projection and sphere.show_components)
        husimi = sphere.husimi() if sphere.show_husimi else []
        controls = sphere.controls() if sphere.show_controls else ""
        sioEmitData = json.dumps({"spin_axis" : sphere.spin_axis(),\
                            "stars" : stuff["stars"],\
                            "state" : sphere.pretty_state(),\
                            "dt" : sphere.dt,\
                            "phase" : phase,\
                            "component_stars" : stuff["component_stars"],\
                            "plane_stars" : stuff["plane_stars"],\
                            "plane_component_stars" : stuff["plane_component_stars"],\
                            "husimi" : husimi,\
                            "controls" : controls});
        sio.emit("animate", sioEmitData)
        sio.sleep(0)

@app.route("/")
def root():
    global thread
    if thread is None:
        thread = sio.start_background_task(animate)
    return render_template("spheres.html")

@app.route("/keypress/")
def key_press():
    global sphere
    keyCode = int(request.args.get('keyCode'))
    stuff = {'success':True, 'collapsed':False}
    if (keyCode == 97):
        sphere.rotate("x", inverse=True)
    elif (keyCode == 100):
        sphere.rotate("x", inverse=False)
    elif (keyCode == 115):
        sphere.rotate("y", inverse=True)
    elif (keyCode == 119):
        sphere.rotate("y", inverse=False)
    elif (keyCode == 122):
        sphere.rotate("z", inverse=True)
    elif (keyCode == 120):
        sphere.rotate("z", inverse=False)
    elif (keyCode == 117):
        sphere.evolving = False if sphere.evolving else True
    elif (keyCode == 105):
        sphere.random_state()
    elif (keyCode == 111):
        sphere.random_energy()
    elif (keyCode == 106):
        sphere.show_projection = False if sphere.show_projection else True
    elif (keyCode == 107):
        sphere.show_components = False if sphere.show_components else True
    elif (keyCode == 108):
        sphere.show_phase = False if sphere.show_phase else True
    elif (keyCode == 91):
        sphere.dt = sphere.dt-0.001
    elif (keyCode == 93):
        sphere.dt = sphere.dt+0.001
    elif (keyCode == 112):
        sphere.show_husimi = False if sphere.show_husimi else True
    elif (keyCode == 113):
        sphere.destroy_star()
    elif (keyCode == 101):
        sphere.create_star()
    elif (keyCode == 46):
        sphere.show_controls = False if sphere.show_controls else True
    elif (keyCode == 49):
        pick, L, probabilities = sphere.collapse(sphere.paulis()[0][0])
        message = "\t%.2f of %s\n\twith {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
        stuff["pick"] = message
        stuff["collapsed"] = True
    elif (keyCode == 50):
        pick, L, probabilities  = sphere.collapse(sphere.paulis()[0][1])
        message = "\t%.2f of %s\n\twith {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
        stuff["pick"] = message
        stuff["collapsed"] = True
    elif (keyCode == 51):
        pick, L, probabilities  = sphere.collapse(sphere.paulis()[0][2])
        message = "\t%.2f of %s\n\twith {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
        stuff["pick"] = message
        stuff["collapsed"] = True
    elif (keyCode == 52):
        if sphere.energy != None:
            pick, L, probabilities = sphere.collapse(sphere.energy)
            message = "\t%.2f of %s\n\twith {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
            stuff["pick"] = message
            stuff["collapsed"] = True
    elif (keyCode == 53):
        pick, L, probabilities  = sphere.collapse(qt.rand_herm(sphere.n()))
        message = "\t%.2f of %s\n\twith {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
        stuff["pick"] = message
        stuff["collapsed"] = True
    return json.dumps(stuff), 200, {'ContentType':'application/json'} 

##################################################################################################################

if __name__ == '__main__':
    import sys
    import eventlet
    import eventlet.wsgi
    app = socketio.Middleware(sio, app)
    port = int(sys.argv[1])
    eventlet.wsgi.server(eventlet.listen(('', port)), app) 
