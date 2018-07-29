from spheres import *
from mixedsphere import *
from puresphere import *

the_spheres = Spheres()
sphere = the_spheres.add_child(qt.rand_ket(2))
new_sphere = None
did_split = None

options = {}

def reset_options():
    global options
    options["show_phase"] = True
    options["show_components"] = False
    options["show_projection"] = False
    options["show_husimi"] = False
    options["show_measurements"] = False
    options["rotate/boost"] = 0
    options["component_unitarity"] = True
    options["click_selected"] = "sphere"
    options["distinguishable_selected"] = "sphere"
    options["symmetrical_selected"] = "sphere"
    options["1d_harmonic_oscillator"] = False
    options["2d_harmonic_oscillator"] = False
    options["show_others"] = True

reset_options()

##################################################################################################################

import os
import sys
import json
import time
import logging
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, request, Response, render_template
    
sio = socketio.Server(async_mode='eventlet')
app = Flask("spheres")
app.debug = True
app.wsgi_app = socketio.Middleware(sio, app.wsgi_app)
thread = None

#log = logging.getLogger('werkzeug')
#log.disabled = True
#app.logger.disabled = True

def animate():
    global the_spheres
    global sphere
    global options
    global new_sphere
    global did_split
    while True:
      #  try:
        if new_sphere != None:
            sphere = the_spheres.finish_up_collide(*new_sphere)
            new_sphere = None
            sphere.refresh()

        if did_split != None:
            sphere = the_spheres.finish_up_split(*did_split)
            did_split = None
            sphere.refresh()

        if sphere.evolving:
            if sphere.evolution == "1D":
                sphere.harmonic_oscillator_1D_evolve()
            elif sphere.evolution == "2D":
                sphere.harmonic_oscillator_2D_evolve()
            elif sphere.evolution == "spin":
                sphere.evolve(sphere.energy)

        phase = sphere.phase() if options["show_phase"] else []

        stuff = sphere.allstars(plane_stars=options["show_projection"],\
                                component_stars=options["show_components"],\
                                plane_component_stars=options["show_projection"]\
                                    and options["show_components"]) \
                    if sphere.pure_sphere != None else \
                            {"stars": [], "plane_stars": [], "component_stars": [], "plane_component_stars": []}

        husimi = sphere.husimi() if options["show_husimi"] else []
        controls = sphere.pretty_measurements(harmonic1D=options["1d_harmonic_oscillator"],\
                                              harmonic2D=options["2d_harmonic_oscillator"])\
                                                if options["show_measurements"] else ""

        sym_arrows = sphere.sym_arrows() if options["symmetrical_selected"] != "sphere" and sphere.pure_sphere != None else []

        piece_arrows = []
        separability = []
        skies = {}
        if sphere.dimensionality != None:
            pieces = sphere.distinguishable_pieces()
            piece_arrows = sphere.dist_pieces_spin(pieces)
            separability = sphere.are_separable(pieces)
            skies = sphere.separable_skies(pieces, separability)

        harmonic_osc_1d = sphere.harmonic_oscillator_1D() if options["1d_harmonic_oscillator"] and sphere.pure_sphere != None else {}
        harmonic_osc_2d = sphere.harmonic_oscillator_2D() if options["2d_harmonic_oscillator"] and sphere.pure_sphere != None else {}

        data = json.dumps({"spin_axis" : sphere.spin_axis(),\
                            "stars" : stuff["stars"],\
                            "state" : sphere.pretty_state(),\
                            "dt" : sphere.dt,\
                            "phase" : phase,\
                            "component_stars" : stuff["component_stars"],\
                            "plane_stars" : stuff["plane_stars"],\
                            "plane_component_stars" : stuff["plane_component_stars"],\
                            "husimi" : husimi,\
                            "controls" : controls,\
                            "piece_arrows" : piece_arrows,\
                            "separability" : separability,\
                            "skies" : skies,\
                            "1d_harmonic_oscillator" : harmonic_osc_1d,\
                            "2d_harmonic_oscillator" : harmonic_osc_2d,\
                            "sym_arrows" : sym_arrows,\
                            "others" : the_spheres.pretty_children(sphere) if options["show_others"] else ""});
        sio.emit("animate", data)
        sio.sleep(0.001)
        #except Exception as e:
        #    print("animate error!: %s" % e)
        #    sys.stdout.flush() 

@app.route("/")
def root():
    global thread
    if thread is None:
        thread = sio.start_background_task(animate)
    return render_template("spheres.html")

@sio.on('connect')
def connect(sid, data):
    print("connected: ", sid)

@sio.on('disconnect')
def disconnect(sid):
    print("disconnected: ", sid)

@sio.on("keypress")
def key_press(sid, data):
    global sphere
    global options
    keyCode = int(data["keyCode"])
    print("keypress: keyCode %d" % keyCode)
    if (keyCode == 97): # 'a'
        do_rotation("x", inverse=True)
    elif (keyCode == 100): # 'd'
        do_rotation("x", inverse=False)
    elif (keyCode == 115): # 's'
        do_rotation("y", inverse=True)
    elif (keyCode == 119): # 'w'
        do_rotation("y", inverse=False)
    elif (keyCode == 122): # 'z'
        do_rotation("z", inverse=True)
    elif (keyCode == 120): # 'x'
        do_rotation("z", inverse=False)
    elif (keyCode == 114): # 'r'
        #options["component_unitarity"] = False if options["component_unitarity"] else True
        options["show_others"] = False if options["show_others"] else True
    elif (keyCode == 117): # 'u'
        sphere.evolving = False if sphere.evolving else True
    elif (keyCode == 105): # 'i'
        if sphere.pure_sphere != None:
            sphere.random_state()
    elif (keyCode == 111): # 'o'
        sphere.random_energy()
    elif (keyCode == 106): # 'j'
        options["show_projection"] = False if options["show_projection"] else True
    elif (keyCode == 107): # 'k'
        options["show_components"] = False if options["show_components"] else True
    elif (keyCode == 108): # 'l'
        options["show_phase"] = False if options["show_phase"] else True
    elif (keyCode == 91): # '['
        sphere.dt = sphere.dt-0.00025
    elif (keyCode == 93): # ']'
        sphere.dt = sphere.dt+0.00025
    elif (keyCode == 112): # 'p'
        options["show_husimi"] = False if options["show_husimi"] else True
    elif (keyCode == 113): # 'q'
        if sphere.pure_sphere != None:
            sphere.destroy_star()
            sio.emit("new_dist_ctrls", {"new_dist_ctrls": ""})
            options["symmetrical_selected"] = "sphere"
            options["distinguishable_selected"] = "sphere"
    elif (keyCode == 101): # 'e'
        if sphere.pure_sphere != None:
            sphere.create_star()
            sio.emit("new_dist_ctrls", {"new_dist_ctrls": ""})
            options["symmetrical_selected"] = "sphere"
            options["distinguishable_selected"] = "sphere"
    elif (keyCode == 45): # '-'
        options["show_measurements"] = False if options["show_measurements"] else True
    elif (keyCode == 102): # 'f'
        do_collapse("x")
    elif (keyCode == 103): # 'g'
        do_collapse("y")
    elif (keyCode == 104): # 'h'
        do_collapse("z")
    elif (keyCode == 121): # 'y'
        do_collapse("h")
    elif (keyCode == 116): # 't'
        do_collapse("r")
    elif (keyCode == 109): # 'm'
        if sphere.pure_sphere != None:
            options["rotate/boost"] += 1
            if options["rotate/boost"] > 2:
                options["rotate/boost"] = 0
            if options["rotate/boost"] == 0:
                sio.emit("collapsed", json.dumps({"message": "spin rotation activated!"}))
            elif options["rotate/boost"] == 1:
                sio.emit("collapsed", json.dumps({"message": "boost activated!"}))
            elif options["rotate/boost"] == 2:
                sio.emit("collapsed", json.dumps({"message": "mobius rotation activated!"}))
    elif (keyCode == 110): # 'n'
        if sphere.pure_sphere != None:
            if options["symmetrical_selected"] == "sphere":
                options["symmetrical_selected"] = 0
            else:
                options["symmetrical_selected"] += 1
                if options["symmetrical_selected"] >= sphere.n()-1:
                    options["symmetrical_selected"] = "sphere"
            if options["symmetrical_selected"] == "sphere":
                sio.emit("collapsed", json.dumps({"message": "%s selected for measurement/rotation" %\
                    (str(options["symmetrical_selected"]))}))        
            else: 
                sio.emit("collapsed", json.dumps({"message": "symmetrical %s selected for measurement/rotation" %\
                    (str(options["symmetrical_selected"]))}))  
    elif (keyCode == 99): # 'c'
        if sphere.pure_sphere != None:
            options["1d_harmonic_oscillator"] = False if options["1d_harmonic_oscillator"] else True
            if options["1d_harmonic_oscillator"]:
                sio.emit("collapsed", json.dumps({"message": "showing 1D harmonic oscillator!"}))
            else:
                sio.emit("collapsed", json.dumps({"message": "hiding 1D harmonic oscillator!"}))
    elif (keyCode == 118): # 'v'   
        if sphere.pure_sphere != None:  
            options["2d_harmonic_oscillator"] = False if options["2d_harmonic_oscillator"] else True
            if options["2d_harmonic_oscillator"]:
                sio.emit("collapsed", json.dumps({"message": "showing 2D harmonic oscillator!"}))
            else:
                sio.emit("collapsed", json.dumps({"message": "hiding 2D harmonic oscillator!"}))
    elif (keyCode == 98): # 'b'
        if sphere.pure_sphere != None:
            if sphere.evolution == "spin":
                sphere.evolution = "1D"
            elif sphere.evolution == "1D":
                sphere.evolution = "2D"
            elif sphere.evolution == "2D":
                sphere.evolution = "spin"
            sio.emit("collapsed", json.dumps({"message": "evolution type: %s!" % sphere.evolution}))
    elif (keyCode == 59): # ';'
        if sphere.pure_sphere != None:
            sphere.harmonic_oscillator_1D_collapse("position")
            sio.emit("collapsed", json.dumps({"message": "1D position collapse!"}))
    elif (keyCode == 39): # '''
        if sphere.pure_sphere != None:
            sphere.harmonic_oscillator_1D_collapse("momentum")
            sio.emit("collapsed", json.dumps({"message": "1D momentum collapse!"}))
    elif (keyCode == 92): # '\'
        if sphere.pure_sphere != None:
            sphere.harmonic_oscillator_1D_collapse("number")
            sio.emit("collapsed", json.dumps({"message": "1D number collapse!"}))
    elif (keyCode == 61): # '='
        if sphere.pure_sphere != None:
            sphere.harmonic_oscillator_1D_collapse("energy")
            sio.emit("collapsed", json.dumps({"message": "1D energy collapse!"}))

def do_rotation(direction, inverse=False):
    global sphere
    global options
    if options["distinguishable_selected"] != "sphere":
        sphere.rotate_distinguishable(options["distinguishable_selected"], direction, dt=sphere.dt, inverse=inverse)
    else:
        if options["symmetrical_selected"] != "sphere":
            sphere.rotate_symmetrical(direction, options["symmetrical_selected"], dt=sphere.dt, inverse=inverse)
        else:
            if options["rotate/boost"] == 0:
                selected = options["click_selected"]
                if selected.startswith("star"):
                    sphere.rotate_star(int(selected[selected.index("_")+1:]), direction, inverse=inverse)
                elif selected.startswith("component"):
                    sphere.rotate_component(int(selected[selected.index("_")+1:]), direction, inverse=inverse,\
                        unitary=options["component_unitarity"])
                else:
                    sphere.rotate(direction, inverse=inverse)
            elif options["rotate/boost"] == 2:
                sphere.mink_rotate(direction, sphere.dt, inverse=inverse)
            elif options["rotate/boost"] == 1:
                sphere.boost(direction, sphere.dt, inverse=inverse)

def do_collapse(direction):
    global sphere
    global options
    if options["distinguishable_selected"] != "sphere":
        if direction != "h":
            sphere.distinguishable_collapse(options["distinguishable_selected"], direction)
            message = "distinguishable %d collapse!" % (options["distinguishable_selected"])
            sio.emit("collapsed", json.dumps({"message": message}))
    else:
        if options["symmetrical_selected"] == "sphere":
            op = None
            if direction == "x":
                op = sphere.paulis()[0][0]
            elif direction == "y":
                op = sphere.paulis()[0][1]
            elif direction == "z":
                op = sphere.paulis()[0][2]
            elif direction == "h":
                op = sphere.energy
            elif direction == "r":
                op = qt.rand_herm(sphere.n())
            pick, L, probabilities = sphere.collapse(op)
            message = "last collapse: %.2f of %s\n                with {%s}!" \
                % (L[pick], np.array_str(L, precision=2, suppress_small=True),\
                        " ".join(["%.2f%%" % (100*p) for p in probabilities]))
            sio.emit("collapsed", json.dumps({"message": message}))
        else:
            sphere.symmetrical_collapse(direction, options["symmetrical_selected"])
            message = "symmetrical %d collapse!" % (options["symmetrical_selected"])
            sio.emit("collapsed", json.dumps({"message": message}))

@sio.on("selected") # click on sphere/white star/red star
def select(sid, data):
    global options
    options["click_selected"] = data["selected"]

@sio.on("dim_set") # dims text box
def dim_set(sid, data):
    global sphere
    global options
    options["distinguishable_selected"] = "sphere"
    if data["dims"].strip() == "":
        sphere.dimensionality = None
        sio.emit("new_dist_ctrls", {"new_dist_ctrls": ""})
    else:
        try:
            dims = [int(d.strip()) for d in data["dims"].split(",")]
            if np.prod(np.array(dims)) == sphere.n() and dims.count(1) == 0:
                sphere.dimensionality = dims
                ctrls = "<div id='dist_ctrls_form'>"
                i = 0
                ctrls += "<input type='radio' name='dist_selected' value='-1'>sphere<br>"
                for d in sphere.dimensionality:
                    ctrls += "<input type='radio' name='dist_selected' value='%d'>%d<br>" % (i, d)
                    i += 1
                ctrls += "</form>"
                sio.emit("new_dist_ctrls", {"new_dist_ctrls": ctrls})
        except:
            print("dim_set error!: %s" % e)
            sys.stdout.flush() 

@sio.on("dim_choice") # distinguishable selection
def dim_choice(sid, data):
    global options
    choice = int(data["choice"])
    if choice == -1:
        options["distinguishable_selected"] = "sphere"
    else:
        options["distinguishable_selected"] = choice

@sio.on("collide")
def collide(sid, data):
    global the_spheres
    global sphere
    global new_sphere
    if data["i"].isdigit():
        i = int(data["i"])
        if i < len(the_spheres.children) and i != the_spheres.children.index(sphere):
            new_sphere = the_spheres.collide_children(sphere, the_spheres.children[i])
            reset_options()
    #print(the_spheres.children)

@sio.on("swap")
def swap(sid, data):
    global the_spheres
    global sphere
    if data["i"].isdigit():
        i = int(data["i"])
        if i < len(the_spheres.children):
            sphere = the_spheres.children[i]
            sphere.refresh()
            reset_options()

@sio.on("create")
def create(sid, data):
    global the_spheres
    the_spheres.add_child(qt.rand_ket(2))

@sio.on("split")
def split(sid, data):
    global the_spheres
    global sphere
    global did_split
    a = float(data["a"])
    b = float(data["b"])
    did_split = the_spheres.split_child(sphere, a, b)
    reset_options()

##################################################################################################################

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    port = int(sys.argv[1])
    eventlet.wsgi.server(eventlet.listen(('', port)), app) 