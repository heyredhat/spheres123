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

#log = logging.getLogger('werkzeug')
#log.disabled = True
#app.logger.disabled = True

selected = "sphere";
distinguishable_selected = "sphere"

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

        piece_arrows = []
        if sphere.dimensionality != None:
            pieces = sphere.distinguishable_pieces()
            piece_arrows = sphere.dist_pieces_spin(pieces)

        sioEmitData = json.dumps({"spin_axis" : sphere.spin_axis(),\
                            "stars" : stuff["stars"],\
                            "state" : sphere.pretty_state(),\
                            "dt" : sphere.dt,\
                            "phase" : phase,\
                            "component_stars" : stuff["component_stars"],\
                            "plane_stars" : stuff["plane_stars"],\
                            "plane_component_stars" : stuff["plane_component_stars"],\
                            "husimi" : husimi,\
                            "controls" : controls,\
                            "piece_arrows": piece_arrows});
        sio.emit("animate", sioEmitData)
        sio.sleep(0)

@sio.on("selected")
def select(sid, data):
    global selected
    selected = data["selected"]

@app.route("/")
def root():
    global thread
    if thread is None:
        thread = sio.start_background_task(animate)
    return render_template("spheres.html")

@sio.on("dim_set")
def dim_set(sid, data):
    global sphere
    if data["dims"].strip() == "":
        sphere.dimensionality = None
        sio.emit("new_dist_ctrls", {"new_dist_ctrls": ""})
    else:
        dims = [int(d.strip()) for d in data["dims"].split(",")]
        if np.prod(np.array(dims)) == sphere.n():
            sphere.dimensionality = dims
        ctrls = "<div id='dist_ctrls_form'>"
        i = 0
        ctrls += "<input type='radio' name='dist_selected' value='-1'>sphere<br>"
        for d in sphere.dimensionality:
            ctrls += "<input type='radio' name='dist_selected' value='%d'>%d<br>" % (i, d)
            i += 1
        ctrls += "</form>"
        sio.emit("new_dist_ctrls", {"new_dist_ctrls": ctrls})

dist_selected = "sphere"

@sio.on("dim_choice")
def dim_choice(sid, data):
    global dist_selected
    choice = int(data["choice"])
    if choice == -1:
        dist_selected = "sphere"
    else:
        dist_selected = choice

mink = 0
unitary_component = True
to_measure = "sphere"
#@app.route("/keypress/")
@sio.on("keypress")
def key_press(sid, data):
    global dist_selected
    global mink
    global unitary_component
    global sphere
    global to_measure
    #keyCode = int(request.args.get('keyCode'))
    keyCode = int(data["keyCode"])
    #stuff = {'success':True, 'collapsed':False}
    #print(keyCode)

    #print(keyCode)
    if (keyCode == 97):
        if dist_selected != "sphere":
            sphere.rotate_distinguishable_piece(dist_selected, "x", dt=sphere.dt, inverse=True)
        else:
            if to_measure != "sphere":
                sphere.boson_rotate("x", to_measure, dt=sphere.dt, inverse=True)
            else:
                if mink == 0:
                    if selected.startswith("star"):
                        sphere.rotate_star(int(selected[selected.index("_")+1:]), "x", inverse=True)
                    elif selected.startswith("component"):
                        sphere.rotate_component(int(selected[selected.index("_")+1:]), "x", inverse=True, unitary=unitary_component)
                    else:
                        sphere.rotate("x", inverse=True)
                #elif mink == 2:
                #    sphere.mink_rotate("x", sphere.dt, inverse=True)
                elif mink == 1:
                    sphere.boost("x", sphere.dt, inverse=True)
    elif (keyCode == 100):  
        if dist_selected != "sphere":
            sphere.rotate_distinguishable_piece(dist_selected, "x", dt=sphere.dt, inverse=False)
        else:
            if to_measure != "sphere":
                sphere.boson_rotate("x", to_measure, dt=sphere.dt, inverse=False)
            else:     
                if mink == 0:     
                    if selected.startswith("star"):
                        sphere.rotate_star(int(selected[selected.index("_")+1:]), "x", inverse=False)
                    elif selected.startswith("component"):
                        sphere.rotate_component(int(selected[selected.index("_")+1:]), "x", inverse=False, unitary=unitary_component)
                    else:
                        sphere.rotate("x", inverse=False)
                #elif mink == 2:
                #    sphere.mink_rotate("x", sphere.dt, inverse=False)
                elif mink == 1:
                    sphere.boost("x", sphere.dt, inverse=False)
    elif (keyCode == 115):
        if dist_selected != "sphere":
            sphere.rotate_distinguishable_piece(dist_selected, "y", dt=sphere.dt, inverse=True)
        else:
            if to_measure != "sphere":
                sphere.boson_rotate("y", to_measure, dt=sphere.dt, inverse=True)
            else:
                if mink == 0:
                    if selected.startswith("star"):
                        sphere.rotate_star(int(selected[selected.index("_")+1:]), "y", inverse=True)
                    elif selected.startswith("component"):
                        sphere.rotate_component(int(selected[selected.index("_")+1:]), "y", inverse=True, unitary=unitary_component)
                    else:
                        sphere.rotate("y", inverse=True)
                #elif mink == 2:
                #    sphere.mink_rotate("y", sphere.dt, inverse=True)
                elif mink == 1:
                    sphere.boost("y", sphere.dt, inverse=True)
    elif (keyCode == 119):
        if dist_selected != "sphere":
            sphere.rotate_distinguishable_piece(dist_selected, "y", dt=sphere.dt, inverse=False)
        else:
            if to_measure != "sphere":
                sphere.boson_rotate("y", to_measure, dt=sphere.dt, inverse=False)
            else:
                if mink == 0:
                    if selected.startswith("star"):
                        sphere.rotate_star(int(selected[selected.index("_")+1:]), "y", inverse=False)
                    elif selected.startswith("component"):
                        sphere.rotate_component(int(selected[selected.index("_")+1:]), "y", inverse=False, unitary=unitary_component)
                    else:
                        sphere.rotate("y", inverse=False)
                #elif mink == 2:
                #    sphere.mink_rotate("y", sphere.dt, inverse=False)
                elif mink == 1:
                    sphere.boost("y", sphere.dt, inverse=False)
    elif (keyCode == 122):
        if dist_selected != "sphere":
            sphere.rotate_distinguishable_piece(dist_selected, "z", dt=sphere.dt, inverse=True)
        else:
            if to_measure != "sphere":
                sphere.boson_rotate("z", to_measure, dt=sphere.dt, inverse=True)
            else:
                if mink == 0:
                    if selected.startswith("star"):
                        sphere.rotate_star(int(selected[selected.index("_")+1:]), "z", inverse=True)
                    elif selected.startswith("component"):
                        sphere.rotate_component(int(selected[selected.index("_")+1:]), "z", inverse=True, unitary=unitary_component)
                    else:
                        sphere.rotate("z", inverse=True)
                #elif mink == 2:
                #    sphere.mink_rotate("z", sphere.dt, inverse=True)
                elif mink == 1:
                    sphere.boost("z", sphere.dt, inverse=True)
    elif (keyCode == 120):
        if dist_selected != "sphere":
            sphere.rotate_distinguishable_piece(dist_selected, "z", dt=sphere.dt, inverse=False)
        else:
            if to_measure != "sphere":
                sphere.boson_rotate("z", to_measure, dt=sphere.dt, inverse=False)
            else:
                if mink == 0:
                    if selected.startswith("star"):
                        sphere.rotate_star(int(selected[selected.index("_")+1:]), "z", inverse=False)
                    elif selected.startswith("component"):
                        sphere.rotate_component(int(selected[selected.index("_")+1:]), "z", inverse=False, unitary=unitary_component)
                    else:
                        sphere.rotate("z", inverse=False)
                #elif mink == 2:
                #    sphere.mink_rotate("z", sphere.dt, inverse=False)
                elif mink == 1:
                    sphere.boost("z", sphere.dt, inverse=False)
    elif (keyCode == 48): # 0
        unitary_component = False if unitary_component else True
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
        sphere.dt = sphere.dt-0.00025
    elif (keyCode == 93):
        sphere.dt = sphere.dt+0.00025
    elif (keyCode == 112):
        sphere.show_husimi = False if sphere.show_husimi else True
    elif (keyCode == 113):
        sphere.destroy_star()
        to_measure = "sphere"
    elif (keyCode == 101):
        sphere.create_star()
        to_measure = "sphere"
    elif (keyCode == 46):
        sphere.show_controls = False if sphere.show_controls else True
    elif (keyCode == 49):
        if dist_selected != "sphere":
            sphere.measure_distinguishable_piece(dist_selected, "x")
            message = "distinguishable %d collapse!" % (dist_selected)
            sio.emit("collapsed", json.dumps({"message": message}))
        else:
            if to_measure == "sphere":
                #running = False
                pick, L, probabilities = sphere.collapse(sphere.paulis()[0][0])
                #message = "\t%.2f of %s\n\twith {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
                message = "last collapse: %.2f of %s\n                with {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
                sio.emit("collapsed", json.dumps({"message": message}))
                #stuff["pick"] = message
                #stuff["collapsed"] = True
            else:
                sphere.boson_collapse("x", to_measure)
                message = "boson %d collapse!" % (to_measure)
                sio.emit("collapsed", json.dumps({"message": message}))
    elif (keyCode == 50):
        if dist_selected != "sphere":
            sphere.measure_distinguishable_piece(dist_selected, "y")
            message = "distinguishable %d collapse!" % (dist_selected)
            sio.emit("collapsed", json.dumps({"message": message}))
        else:
            if to_measure == "sphere":
                #running = False
                pick, L, probabilities  = sphere.collapse(sphere.paulis()[0][1])
                message = "last collapse: %.2f of %s\n                with {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
                sio.emit("collapsed", json.dumps({"message": message}))
                #stuff["pick"] = message
                #stuff["collapsed"] = True
            else:
                sphere.boson_collapse("y", to_measure)
                message = "boson %d collapse!" % (to_measure)
                sio.emit("collapsed", json.dumps({"message": message}))
    elif (keyCode == 51):
        if dist_selected != "sphere":
            sphere.measure_distinguishable_piece(dist_selected, "z")
            message = "distinguishable %d collapse!" % (dist_selected)
            sio.emit("collapsed", json.dumps({"message": message}))
        else:
            if to_measure == "sphere":
                #running = False
                pick, L, probabilities  = sphere.collapse(sphere.paulis()[0][2])
                message = "%.2f of %s\n                with {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
                sio.emit("last collapse: collapsed", json.dumps({"message": message}))
                #stuff["pick"] = message
                #stuff["collapsed"] = True
            else:
                sphere.boson_collapse("z", to_measure)
                message = "boson %d collapse!" % (to_measure)
                sio.emit("collapsed", json.dumps({"message": message}))
    elif (keyCode == 52):
        if dist_selected != "sphere":
            pass
        else:
            if to_measure == "sphere":
                if sphere.energy != None:
                    #running = False
                    pick, L, probabilities = sphere.collapse(sphere.energy)
                    message = "%.2f of %s\n                with {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
                    sio.emit("last collapse: collapsed", json.dumps({"message": message}))
                    #stuff["pick"] = message
                    #stuff["collapsed"] = True
    elif (keyCode == 53):
        if dist_selected != "sphere":
            sphere.measure_distinguishable_piece(dist_selected, "r")
            message = "distinguishable %d collapse!" % (dist_selected)
            sio.emit("collapsed", json.dumps({"message": message}))
        else:
            if to_measure == "sphere":
                #running = False
                pick, L, probabilities  = sphere.collapse(qt.rand_herm(sphere.n()))
                message = "%.2f of %s\n                with {%s}!" % (L[pick], np.array_str(L, precision=2, suppress_small=True), " ".join(["%.2f%%" % (100*p) for p in probabilities]))
                sio.emit("last collapse: collapsed", json.dumps({"message": message}))
                #stuff["pick"] = message
                #stuff["collapsed"] = True
            else:
                sphere.boson_collapse("r", to_measure)
                message = "boson %d collapse!" % (to_measure)
                sio.emit("collapsed", json.dumps({"message": message}))
    elif (keyCode == 109): # m
        mink += 1
        if mink > 1:
            mink = 0
    elif (keyCode == 110): # n
        if to_measure == "sphere":
            to_measure = 0
        else:
            to_measure += 1
            if to_measure >= sphere.n()-1:
                to_measure = "sphere"
        if to_measure == "sphere":
            sio.emit("collapsed", json.dumps({"message": "%s selected for measurement/rotation" % (str(to_measure))}))        
        else: 
            sio.emit("collapsed", json.dumps({"message": "boson %s selected for measurement/rotation" % (str(to_measure))}))        
    #return json.dumps(stuff), 200, {'ContentType':'application/json'} 

@sio.on("start")
def start(sid):
    global running
    print("starting...")
    running = True
    #return json.dumps({"success": True}), 200, {'ContentType':'application/json'} 

@sio.on("stop")
def stop(sid):
    global running
    print("stopping...")
    running = False
    #return json.dumps({"success": True}), 200, {'ContentType':'application/json'} 

##################################################################################################################

if __name__ == '__main__':
    import sys
    import eventlet
    import eventlet.wsgi
    app = socketio.Middleware(sio, app)
    port = int(sys.argv[1])
    eventlet.wsgi.server(eventlet.listen(('', port)), app) 
