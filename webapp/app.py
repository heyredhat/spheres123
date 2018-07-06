from flask import Flask, request, Response, render_template
import logging
import json

from spheres import *

sphere = Sphere()
sphere.state = qt.basis(2,0)

app = Flask("spheres")
#log = logging.getLogger('werkzeug')
#log.disabled = True
#app.logger.disabled = True

@app.route("/")
def root():
    return render_template("spheres.html")

@app.route("/animate/")
def animate():
    global sphere
    sphere.update()
    husimi = sphere.husimi() if sphere.calculate_husimi else []
    phase = sphere.phase() if sphere.show_phase else []
    return Response(json.dumps({"spin_axis" : sphere.spin_axis(),\
                                "stars" : sphere.stars(),\
                                "state" : sphere.pretty_state(),\
                                "dt" : sphere.dt,\
                                "phase" : phase,\
                                "husimi" : husimi}),\
                    mimetype="application/json")

@app.route("/keypress/")
def key_press():
    global dt
    global state
    keyCode = int(request.args.get('keyCode'))
    print(keyCode)
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
    elif (keyCode == 112) :
        sphere.calculate_husimi = False if sphere.calculate_husimi else True
    elif (keyCode == 108):
        sphere.show_phase = False if sphere.show_phase else True
    elif (keyCode == 91):
        sphere.dt = sphere.dt-0.001
    elif (keyCode == 93):
        sphere.dt = sphere.dt+0.001
    elif (keyCode == 113):
        sphere.destroy_star()
    elif (keyCode == 101):
        sphere.create_star()
    return Response()