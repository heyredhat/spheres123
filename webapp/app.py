import math
import qutip as qt

from flask import Flask, request, Response, render_template
from http.server import * 
import json

def evolve(state, operator, dt=0.01, inverse=False):
    unitary = (-2*math.pi*1j*operator*dt).expm()
    if inverse:
        unitary = unitary.dag()
    return unitary*state

dt = 0.01
state = qt.rand_ket(2)

app = Flask("spheres")

@app.route("/")
def root():
    return render_template("spheres.html")

@app.route("/animate/")
def animate():
    global state
    return Response(json.dumps({ "x" : qt.expect(qt.sigmax(), state),\
                                 "y" : qt.expect(qt.sigmay(), state),\
                                 "z" : qt.expect(qt.sigmaz(), state)}),\
                    mimetype="application/json")

@app.route("/keypress/")
def key_press():
    global dt
    global state
    keyCode = int(request.args.get('keyCode'))
    if (keyCode == 97):
        state = evolve(state, qt.sigmax(), dt, True)  # a : X-
    elif (keyCode == 100):
        state = evolve(state, qt.sigmax(), dt, False) # d : X+
    elif (keyCode == 115):
        state = evolve(state, qt.sigmay(), dt, True)  # s : Y-
    elif (keyCode == 119):
        state = evolve(state, qt.sigmay(), dt, False) # w : Y+
    elif (keyCode == 122):
        state = evolve(state, qt.sigmaz(), dt, True)  # z : Z-
    elif (keyCode == 120):
        state = evolve(state, qt.sigmaz(), dt, False) # x : Z+
    return Response()

if __name__ == '__main__':
    app.run(host='0.0.0.0')