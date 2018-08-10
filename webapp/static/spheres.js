var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
var camera_controls = new THREE.OrbitControls(camera, renderer.domElement);
camera.position.z = 2;

window.addEventListener('resize', function (event) {
	renderer.setSize(window.innerWidth, window.innerHeight);
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();

});

var light = new THREE.AmbientLight(0xffffff);
scene.add(light);

var sphere_geometry = new THREE.SphereGeometry(1, 32, 32);
var sphere_material = new THREE.MeshPhongMaterial({color: 0x0000ff,  transparent: true});
var sphere = new THREE.Mesh(sphere_geometry, sphere_material);
sphere_material.opacity = 0.5;
scene.add(sphere);

var up_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, 0xffff00, 0.1);
scene.add(up_arrow); 

var XY_plane_geometry = new THREE.PlaneGeometry( 7, 7, 7 );
var XY_plane_material = new THREE.MeshBasicMaterial( {color: 0xf000ff, side: THREE.DoubleSide} );
var XY_plane = new THREE.Mesh(XY_plane_geometry, XY_plane_material);
XY_plane_material.transparent = true;
XY_plane_material.opacity = 0.2;
scene.add(XY_plane);

var spin_axis_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, 0xFF0000, 0.1);
scene.add(spin_axis_arrow);

var phase_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, 0x008000, 0.5);
scene.add(phase_arrow);

var stars = [];
var component_stars = [];
var plane_stars = [];
var plane_arrows = [];
var component_plane_stars = [];
var component_plane_arrows = [];
var husimi_pts = false;
var piece_arrows = [];
var piece_spheres = [];
var skies = {};
var harmonic_osc1D = {};
var harmonic_osc2D = {};
var sym_arrows = [];
var iterated_majorana_spheres = [];
var iterated_majorana_stars = [];


/************************************************************************************************************/

var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();
var selected = "sphere";

document.addEventListener( 'mousedown', 
	function ( event ) {
		//event.preventDefault();
		mouse.x = ( event.clientX / renderer.domElement.clientWidth ) * 2 - 1;
		mouse.y = - ( event.clientY / renderer.domElement.clientHeight ) * 2 + 1;
		raycaster.setFromCamera( mouse, camera );
		objects = [sphere].concat(stars).concat(component_stars);
		var intersections = raycaster.intersectObjects(objects);
		if (intersections.length > 0) {
			object = intersections[0].object;
			if (object == sphere) {
				selected = "sphere";
			} else {
				star_index = stars.indexOf(object);
				if (star_index != -1) {
					selected = "star_" + star_index.toString();
				} else {
					component_index = component_stars.indexOf(object);
					if (component_index != -1) {
						selected = "component_" + component_index.toString();
					} else {
						selected = "sphere";
					}
				}
			}
		}
	}, false );

/************************************************************************************************************/

var spheresSocket = io.connect(null, {port: location.port, rememberTransport: false});

spheresSocket.on("animate", function(socketData) {
	spheresSocket.emit("selected", {"selected": selected});
	render(JSON.parse(socketData));
});

spheresSocket.on("collapsed", function(socketData) {
	document.getElementById("last_collapse").innerHTML = JSON.parse(socketData)["message"];
});

spheresSocket.on("new_dist_ctrls", function(socketData) {
	document.getElementById("density_selector").innerHTML = socketData["new_dist_ctrls"];
});

spheresSocket.on("angles", function(socketData) {
	document.getElementById("angles").innerHTML = socketData["angles"];
});

function set_dims() {
	spheresSocket.emit("dim_set", {"dims": document.getElementById("dims").value});
}

function set_density_selected() {
	if (document.querySelector('input[name = "dist_selected"]:checked') != null) {
		spheresSocket.emit("dim_choice", {"choice": 
			document.querySelector('input[name = "dist_selected"]:checked').value});
	}
}

function do_collide() {
	spheresSocket.emit("collide", {"i": document.getElementById("collider").value});
}

function do_swap() {
	spheresSocket.emit("swap", {"i": document.getElementById("swapper").value});
}

function do_create() {
	spheresSocket.emit("create", {"": ''});
}

function do_split() {
	spheresSocket.emit("split", {"a": document.getElementById("split1").value,
							     "b": document.getElementById("split2").value});
}

function do_angles() {
	spheresSocket.emit("penrose", {"": ""});
}

var show_spin_cam = false;
var cam_touched = false;
function spin_cam() {
	if (show_spin_cam == false) {
		show_spin_cam = true;
		sphere.material.side = THREE.BackSide;
	} else {
		show_spin_cam = false;
		sphere.material.side = THREE.FrontSide;
		cam_touched = true;
	}
}

function tunes() {
	if (document.getElementById("sphurs_audio").paused) { 
		document.getElementById("sphurs_audio").play(); 
	} else { 
		document.getElementById("sphurs_audio").pause();
	}
}

function iterated_majorana() {
	spheresSocket.emit("iterated_majorana", {"": ""});
}

/************************************************************************************************************/

document.addEventListener("keypress", function (event) {
	var keyCode = event.which;
	if (keyCode == 47) {
		help_pane = document.getElementById("help");
    	help_pane.style.display = help_pane.style.display == "none" ? "block" : "none";
    } else if (keyCode == 96) {
    	status_pane = document.getElementById("status");
    	status_pane.style.display = status_pane.style.display == "none" ? "block" : "none";
    } else {
    	if (keyCode == 113 || keyCode == 101) {
    		selected = "sphere";
    	} 
    	spheresSocket.emit("keypress", {"keyCode": keyCode});
	}
});

/************************************************************************************************************/

function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}

function render (response) {
	var new_spin_axis = response["spin_axis"];
	var new_stars = response["stars"];
	var new_state = response["state"];
	var new_dt = response["dt"];
	var new_phase = response["phase"];
	var new_component_stars = response["component_stars"];
	var new_plane_stars = response["plane_stars"];
	var new_component_plane_stars = response["plane_component_stars"];
	var new_husimi = response["husimi"];
	var new_controls = response["controls"];
	var new_piece_arrows = response["piece_arrows"];
	var new_separability = response["separability"];
	var new_skies = response["skies"];
	var new_harmonic_osc1D = response["1d_harmonic_oscillator"];
	var new_harmonic_osc2D = response["2d_harmonic_oscillator"];
	var new_sym_arrows = response["sym_arrows"];
	var new_others = response["others"];
	var new_purity = response["pure"];
	var new_iterated_majorana_available = response["iterated_majorana_available"];
	var new_iterated_majorana = response["iterated_majorana"];

	// Update iterated majorana available button
	if (new_iterated_majorana_available == true) {
		maj_button = document.getElementById("maj");
    	maj_button.style.display = "block";
	} else {
		maj_button = document.getElementById("maj");
    	maj_button.style.display = "none";
	}

	// Updated iterated majorana
	if (new_iterated_majorana.length != 0) {
		if (iterated_majorana_stars.length == 0) {
			for (i = 0; i < new_iterated_majorana.length; ++i) {
				var sgeo = new THREE.SphereGeometry(i+2, 32, 32);
				var ccc = Math.random() * 0xffffff
				var smat = new THREE.MeshPhongMaterial({color: ccc,  transparent: true}); 
				var sph = new THREE.Mesh(sgeo, smat); 
				smat.opacity = 0.5;
				smat.side = THREE.BackSide;
				sph.position.set(0,0,0);
				scene.add(sph);
				iterated_majorana_spheres.push(sph);

				var some_new_stars = [];
				for (j = 0; j < new_iterated_majorana[i].length; ++j) {
					var star_geometry = new THREE.SphereGeometry(0.075*(i+2), 32, 32);
					var star_material = new THREE.MeshPhongMaterial({color: ccc});
					var star = new THREE.Mesh(star_geometry, star_material);
					star.position.set((i+2)*new_iterated_majorana[i][j][0], (i+2)*new_iterated_majorana[i][j][1], (i+2)*new_iterated_majorana[i][j][2]);
					some_new_stars.push(star);
					scene.add(star);
				}
				iterated_majorana_stars.push(some_new_stars);
			}
		} else {
 			for (i = 0; i < new_iterated_majorana.length; ++i) {
 				for (j = 0; j < new_iterated_majorana[i].length; ++j) {
 					iterated_majorana_stars[i][j].position.set((i+2)*new_iterated_majorana[i][j][0], (i+2)*new_iterated_majorana[i][j][1], (i+2)*new_iterated_majorana[i][j][2]);
 				}
 			}
		}
	} else {
		for (i = 0; i < iterated_majorana_stars.length; ++i) {
			for (j = 0; j < iterated_majorana_stars[i].length; ++j) {
				scene.remove(iterated_majorana_stars[i][j]);
			}
		}
		for (i = 0; i < iterated_majorana_spheres.length; ++i) {
			scene.remove(iterated_majorana_spheres[i]);
		}
		iterated_majorana_stars = [];
		iterated_majorana_spheres = [];
	}

	// Update mixed/pure
	if (new_purity == true) {
		sphere.material.color.setHex(0x0000ff);
	} else {
		sphere.material.color.setHex(0xd3d3d3);
	}

	// Update others pane
	if (new_others == "") {
    	others_pane = document.getElementById("others");
    	others_pane.style.display = "none";
	} else {
		others_pane = document.getElementById("others");
    	others_pane.style.display = "block";
		document.getElementById("thrs").innerHTML = response["others"];
	}

	// Update arrows corresponding to symmetrical qubits
	if (new_sym_arrows.length == 0) {
		for (i = 0; i < sym_arrows.length; ++i) {
			scene.remove(sym_arrows[i]);
			delete sym_arrows[i];
		}
		sym_arrows = [];
	} else {
		if (sym_arrows.length == 0) {
			for (i = 0; i < new_sym_arrows.length; ++i) {
				axis = new THREE.Vector3(new_sym_arrows[i][0][0], new_sym_arrows[i][0][1], 
										new_sym_arrows[i][0][2]);
				length = new_sym_arrows[i][1];
				var sym_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0xffffff, 1, 1);
				sym_arrow.setDirection(axis);
				sym_arrow.setLength(length);
				sym_arrows.push(sym_arrow);
				scene.add(sym_arrow);
			}
		} else {
			for (i = 0; i < new_sym_arrows.length; ++i) {
				axis = new THREE.Vector3(new_sym_arrows[i][0][0], new_sym_arrows[i][0][1], 
										new_sym_arrows[i][0][2]);
				length = new_sym_arrows[i][1];
				sym_arrows[i].setDirection(axis);
				sym_arrows[i].setLength(length);
			}
		}
	}

	// Update harmonic oscillator 1D
	// ["position"], ["momentum"], ["number"], ["energy"]
	if (Object.keys(new_harmonic_osc1D).length == 0) {
		// delete
		scene.remove(harmonic_osc1D["pos_sphere"]);
		//scene.remove(harmonic_osc1D["pos_arrow"]);
		scene.remove(harmonic_osc1D["mom_arrow"]);
		delete harmonic_osc1D["pos_sphere"];
		//delete harmonic_osc1D["pos_arrow"];
		delete harmonic_osc1D["mom_arrow"];
		harmonic_osc1D = {};
	} else {
		if (Object.keys(harmonic_osc1D).length == 0) {
			// create
			energy = new_harmonic_osc1D["energy"];
			var pos1D_sphere_geometry = new THREE.SphereGeometry(0.05, 32, 32);
			var pos1D_sphere_material = new THREE.MeshPhongMaterial({color: 0xff6ec7 });
			var pos1D_sphere = new THREE.Mesh(pos1D_sphere_geometry, pos1D_sphere_material);
			//var pos1D_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0), 1, color);
			var mom1D_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0x00FF00, 1, 1);

			pos_axis = new THREE.Vector3(new_harmonic_osc1D["position"][0][0], new_harmonic_osc1D["position"][0][1], 
										new_harmonic_osc1D["position"][0][2]);
			pos_length = new_harmonic_osc1D["position"][1];
			pos1D_sphere.position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);
			//pos1D_arrow.setDirection(pos_axis);
			//pos1D_arrow.setLength(pos_length);
			mom_axis = new THREE.Vector3(new_harmonic_osc1D["momentum"][0][0], new_harmonic_osc1D["momentum"][0][1], 
										new_harmonic_osc1D["momentum"][0][2]);
			mom_length = new_harmonic_osc1D["momentum"][1];
			mom1D_arrow.setDirection(mom_axis);
			mom1D_arrow.setLength(mom_length);
			mom1D_arrow.position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);

			scene.add(pos1D_sphere);
			//scene.add(pos1D_arrow);
			scene.add(mom1D_arrow);
			harmonic_osc1D["pos_sphere"] = pos1D_sphere;
			//harmonic_osc1D["pos_arrow"] = pos1D_arrow;
			harmonic_osc1D["mom_arrow"] = mom1D_arrow;
		} else {
			// update
			energy = new_harmonic_osc1D["energy"];
			pos_axis = new THREE.Vector3(new_harmonic_osc1D["position"][0][0], new_harmonic_osc1D["position"][0][1], 
										new_harmonic_osc1D["position"][0][2]);
			pos_length = new_harmonic_osc1D["position"][1];
			mom_axis = new THREE.Vector3(new_harmonic_osc1D["momentum"][0][0], new_harmonic_osc1D["momentum"][0][1], 
										new_harmonic_osc1D["momentum"][0][2]);
			mom_length = new_harmonic_osc1D["momentum"][1];
			harmonic_osc1D["pos_sphere"].position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);
			//harmonic_osc1D["pos_sphere"].material.color = color;
			//harmonic_osc1D["pos_arrow"].setDirection(pos_axis);
			//harmonic_osc1D["pos_arrow"].setLength(pos_length);
			//harmonic_osc1D["pos_arrow"].setColor(color);
			harmonic_osc1D["mom_arrow"].setDirection(mom_axis);
			harmonic_osc1D["mom_arrow"].setLength(mom_length);
			harmonic_osc1D["mom_arrow"].position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);
		}
	}

	// Update harmonic oscillator 2D
	// ["position"], ["momentum"], ["angmo"], ["number"], ["energy"]
	if (Object.keys(new_harmonic_osc2D).length == 0) {
		scene.remove(harmonic_osc2D["pos_sphere"]);
		//scene.remove(harmonic_osc2D["pos_arrow"]);
		scene.remove(harmonic_osc2D["mom_arrow"]);
		scene.remove(harmonic_osc2D["angmo_arrow"]);
		delete harmonic_osc2D["pos_sphere"];
		//delete harmonic_osc2D["pos_arrow"];
		delete harmonic_osc2D["mom_arrow"];
		delete harmonic_osc2D["angmo_arrow"];
		harmonic_osc2D = {};
	} else {
		if (Object.keys(harmonic_osc2D).length == 0) {
			energy = new_harmonic_osc2D["energy"];
			var pos2D_sphere_geometry = new THREE.SphereGeometry(0.05, 32, 32);
			var pos2D_sphere_material = new THREE.MeshPhongMaterial({color: 0x663399 });
			var pos2D_sphere = new THREE.Mesh(pos2D_sphere_geometry, pos2D_sphere_material);
			//var pos2D_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0), 1, color);
			var mom2D_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0x00FF00, 1, 1);
			var angmo_arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0xffa500, 1, 1);

			pos_axis = new THREE.Vector3(new_harmonic_osc2D["position"][0][0], new_harmonic_osc2D["position"][0][1], 
										new_harmonic_osc2D["position"][0][2]);
			pos_length = new_harmonic_osc2D["position"][1];
			pos2D_sphere.position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);
			//pos2D_arrow.setDirection(pos_axis);
			//pos2D_arrow.setLength(pos_length);

			mom_axis = new THREE.Vector3(new_harmonic_osc2D["momentum"][0][0], new_harmonic_osc2D["momentum"][0][1], 
										new_harmonic_osc2D["momentum"][0][2]);
			mom_length = new_harmonic_osc2D["momentum"][1];
			mom2D_arrow.setDirection(mom_axis);
			mom2D_arrow.setLength(mom_length);
			mom2D_arrow.position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);

			angmo_axis = new THREE.Vector3(new_harmonic_osc2D["angmo"][0][0], new_harmonic_osc2D["angmo"][0][1], 
										new_harmonic_osc2D["angmo"][0][2]);
			angmo_length = new_harmonic_osc2D["angmo"][1];
			angmo_arrow.setDirection(angmo_axis);
			angmo_arrow.setLength(angmo_length);
			angmo_arrow.position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);

			scene.add(pos2D_sphere);
			//scene.add(pos2D_arrow);
			scene.add(mom2D_arrow);
			scene.add(angmo_arrow);
			harmonic_osc2D["pos_sphere"] = pos2D_sphere;
			//harmonic_osc2D["pos_arrow"] = pos2D_arrow;
			harmonic_osc2D["mom_arrow"] = mom2D_arrow;
			harmonic_osc2D["angmo_arrow"] = angmo_arrow
		} else {
			energy = new_harmonic_osc2D["energy"];
			pos_axis = new THREE.Vector3(new_harmonic_osc2D["position"][0][0], new_harmonic_osc2D["position"][0][1], 
										new_harmonic_osc2D["position"][0][2]);
			pos_length = new_harmonic_osc2D["position"][1];
			mom_axis = new THREE.Vector3(new_harmonic_osc2D["momentum"][0][0], new_harmonic_osc2D["momentum"][0][1], 
										new_harmonic_osc2D["momentum"][0][2]);
			mom_length = new_harmonic_osc2D["momentum"][1];
			angmo_axis = new THREE.Vector3(new_harmonic_osc2D["angmo"][0][0], new_harmonic_osc2D["angmo"][0][1], 
										new_harmonic_osc2D["angmo"][0][2]);
			angmo_length = new_harmonic_osc2D["angmo"][1];

			harmonic_osc2D["pos_sphere"].position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);
			//harmonic_osc2D["pos_sphere"].material.color = color;
			//harmonic_osc2D["pos_arrow"].setDirection(pos_axis);
			//harmonic_osc2D["pos_arrow"].setLength(pos_length);
			//harmonic_osc2D["pos_arrow"].setColor(color);
			harmonic_osc2D["mom_arrow"].setDirection(mom_axis);
			harmonic_osc2D["mom_arrow"].setLength(mom_length);
			harmonic_osc2D["mom_arrow"].position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);
			harmonic_osc2D["angmo_arrow"].setDirection(angmo_axis);
			harmonic_osc2D["angmo_arrow"].setLength(angmo_length);
			harmonic_osc2D["angmo_arrow"].position.set(pos_length*pos_axis.x, pos_length*pos_axis.y, pos_length*pos_axis.z);
		}
	}

	// Update distinguishable skies
	if (Object.keys(new_skies).length == 0) {
		for (var i in skies) {
			for (j = 0; j < skies[i].length; ++j) {
				scene.remove(skies[i][j]);
			}
			delete skies[i];
		}
		skies = {};
	}

	// Update distinguishable pieces
	if (new_piece_arrows.length == piece_arrows.length) {
		for(i = 0; i < piece_arrows.length; ++i) {
			axis = new THREE.Vector3(new_piece_arrows[i][0][0], new_piece_arrows[i][0][1], 
										new_piece_arrows[i][0][2])
			length = new_piece_arrows[i][1];
			piece_arrows[i].setDirection(axis);
			piece_arrows[i].setLength(length);
			piece_spheres[i].position.set(length*axis.x, length*axis.y, length*axis.z);
			if (new_separability[i] == true) {
				if (i in skies) {
					for (j = 0; j < skies[i].length; ++j) {
						x = length*axis.x + new_skies[i][j][0]*0.1;
						y = length*axis.y + new_skies[i][j][1]*0.1;
						z = length*axis.z + new_skies[i][j][2]*0.1;
						skies[i][j].position.set(x, y, z);
					}
				} else {
					skies[i] = [];
					for (j = 0; j < new_skies[i].length; ++j) {
						var star_geometry = new THREE.SphereGeometry(0.04, 32, 32);
						var star_material = new THREE.MeshPhongMaterial({color: 0xffffff});
						var star = new THREE.Mesh(star_geometry, star_material);
						x = length*axis.x + new_skies[i][j][0]*0.1;
						y = length*axis.y + new_skies[i][j][1]*0.1;
						z = length*axis.z +  new_skies[i][j][2]*0.1;
						star.position.set(x, y, z);
						skies[i].push(star);
						scene.add(star);
					}
				}
			} else {
				if (i in skies) {
					for (j = 0; j < skies[i].length; ++j) {
						scene.remove(skies[i][j]);
					}
					delete skies[i];
				}
			}
		}
	} else {
		for(i = 0; i < piece_arrows.length; ++i) {
			scene.remove(piece_arrows[i]);
			scene.remove(piece_spheres[i]);
		}
		piece_arrows = [];
		piece_spheres = [];
		for(i = 0; i < new_piece_arrows.length; ++i) {
			axis = new THREE.Vector3(new_piece_arrows[i][0][0], new_piece_arrows[i][0][1], 
										new_piece_arrows[i][0][2])
			length = new_piece_arrows[i][1];
			color = Math.random() * 0xffffff
			var arrow = new THREE.ArrowHelper(axis, new THREE.Vector3(0, 0, 0), length, color);
			piece_arrows.push(arrow);
			scene.add(arrow);
			var density_geometry = new THREE.SphereGeometry(0.1, 32, 32);
			var density_material = new THREE.MeshPhongMaterial({color: color});
			var density = new THREE.Mesh(density_geometry, density_material);
			density.position.set(length*axis.x, length*axis.y, length*axis.z);
			piece_spheres.push(density);
			scene.add(density);
		}
	}

	// Update spin axis arrow
	spin_axis_arrow.setDirection(new THREE.Vector3(new_spin_axis[0][0], new_spin_axis[0][1], 
													new_spin_axis[0][2]));
	spin_axis_arrow.setLength(new_spin_axis[1]);

	// Update spin cam
	if (show_spin_cam == true) {
		camera.position.set(new_spin_axis[1]*new_spin_axis[0][0], new_spin_axis[1]*new_spin_axis[0][1], new_spin_axis[1]*new_spin_axis[0][2]);
		//camera.position.set(3*new_spin_axis[0][0], 3*new_spin_axis[0][1], 3*new_spin_axis[0][2]);

		//camera.up = new THREE.Vector3();
		camera.lookAt(new THREE.Vector3(0,0,0));

		up_arrow.visible = false;
	} else {
		if (cam_touched == true) {
			camera.position.set(0, 0, 2);
			camera.lookAt(new THREE.Vector3(0,0,0));
			cam_touched = false;
			up_arrow.visible = true;
		}
	}

	// Update phase arrow
	if (new_phase.length == 0) {
		phase_arrow.visible = false;
	} else {
		phase_arrow.visible = true;
		if (show_spin_cam == true) {
			original_up = new THREE.Vector3(0,0,1);
			new_up = new THREE.Vector3(new_spin_axis[0][0], new_spin_axis[0][1], new_spin_axis[0][2]);
			var quaternion = new THREE.Quaternion(); 
			quaternion.setFromUnitVectors(original_up, new_up);

			phase_dir = new THREE.Vector3(new_phase[0], new_phase[1], 0);
			phase_dir.applyQuaternion(quaternion);
			phase_arrow.setDirection(phase_dir);
		} else {
			phase_arrow.setDirection(new THREE.Vector3(new_phase[0], new_phase[1], 0));
		}
	}

	// Update stars
	if (new_stars.length == stars.length) {
		for(i = 0; i < stars.length; ++i) {
			stars[i].position.set(new_stars[i][0], new_stars[i][1], new_stars[i][2]);
		}
	} else {
		for(i = 0; i < stars.length; ++i) {
			scene.remove(stars[i]);
		}
		stars = [];
		for(i = 0; i < new_stars.length; ++i) {
			var star_geometry = new THREE.SphereGeometry(0.1, 32, 32);
			var star_material = new THREE.MeshPhongMaterial({color: 0xffffff});
			var star = new THREE.Mesh(star_geometry, star_material);
			star.position.set(new_stars[i][0], new_stars[i][1], new_stars[i][2]);
			stars.push(star);
			scene.add(star);
		}
	}

	// Update component stars
	if (new_component_stars.length == component_stars.length) {
		for(i = 0; i < component_stars.length; ++i) {
			component_stars[i].position.set(new_component_stars[i][0], new_component_stars[i][1], 
												new_component_stars[i][2]);
		}
	} else {
		for(i = 0; i < component_stars.length; ++i) {
			scene.remove(component_stars[i]);
		}
		component_stars = []
		for(i = 0; i < new_component_stars.length; ++i) {
			var star_geometry = new THREE.SphereGeometry(0.1, 32, 32);
			var star_material = new THREE.MeshPhongMaterial({color: 0xff0000});
			var star = new THREE.Mesh(star_geometry, star_material);
			star.position.set(new_component_stars[i][0], new_component_stars[i][1], 
								new_component_stars[i][2]);
			component_stars.push(star);
			scene.add(star);
		}
	}

	// Update plane
	if (new_plane_stars.length != 0 || new_component_plane_stars.length != 0) {
		XY_plane.visible = true;
	} else {
		XY_plane.visible = false;
	}

	// Update plane stars
	if (new_plane_stars.length == plane_stars.length) {
		for(i = 0; i < plane_stars.length; ++i) {
			plane_stars[i].position.set(new_plane_stars[i][0], new_plane_stars[i][1], 
											new_plane_stars[i][2]);

			plane_arrows[i].geometry.vertices[1].x = new_plane_stars[i][0];
			plane_arrows[i].geometry.vertices[1].y = new_plane_stars[i][1];
			plane_arrows[i].geometry.vertices[1].z = new_plane_stars[i][2];

			plane_arrows[i].geometry.vertices[2].x = new_stars[i][0];
			plane_arrows[i].geometry.vertices[2].y = new_stars[i][1];
			plane_arrows[i].geometry.vertices[2].z = new_stars[i][2];
			plane_arrows[i].geometry.verticesNeedUpdate = true;
		}
	} else {
		for(i = 0; i < plane_stars.length; ++i) {
			scene.remove(plane_stars[i]);
			scene.remove(plane_arrows[i]);
		}
		plane_stars = [];
		plane_arrows = [];
		for(i = 0; i < new_plane_stars.length; ++i) {
			var star_geometry = new THREE.SphereGeometry(0.05, 32, 32);
			var star_material = new THREE.MeshPhongMaterial({color: 0xffffff});
			var star = new THREE.Mesh(star_geometry, star_material);
			star.position.set(new_plane_stars[i][0], new_plane_stars[i][1], new_plane_stars[i][2]);
			plane_stars.push(star);
			scene.add(star);
			var arrow_geometry = new THREE.Geometry();
			arrow_geometry.dynamic = true;
			arrow_geometry.vertices.push(new THREE.Vector3(0,0,1));
			arrow_geometry.vertices.push(new THREE.Vector3(new_plane_stars[i][0], new_plane_stars[i][1], 
																new_plane_stars[i][2]));
			arrow_geometry.vertices.push(new THREE.Vector3(new_stars[i][0], new_stars[i][1], new_stars[i][2]));
			var arrow_material = new THREE.LineBasicMaterial({
					color: 0xffffff,
					opacity: 0.9,
					linewidth: 2
			});
			var arrow = new THREE.Line(arrow_geometry, arrow_material);
			arrow.position.set(0,0,0);
			plane_arrows.push(arrow);
			scene.add(arrow);
		}
	}

	// Update component plane stars
	if (new_component_plane_stars.length == component_plane_stars.length) {
		for(i = 0; i < component_plane_stars.length; ++i) {
			component_plane_stars[i].position.set(new_component_plane_stars[i][0], 
				new_component_plane_stars[i][1], new_component_plane_stars[i][2]);
			component_plane_arrows[i].geometry.vertices[1].x = new_component_plane_stars[i][0];
			component_plane_arrows[i].geometry.vertices[1].y = new_component_plane_stars[i][1];
			component_plane_arrows[i].geometry.vertices[1].z = new_component_plane_stars[i][2];
			component_plane_arrows[i].geometry.vertices[2].x = new_component_stars[i][0];
			component_plane_arrows[i].geometry.vertices[2].y = new_component_stars[i][1];
			component_plane_arrows[i].geometry.vertices[2].z = new_component_stars[i][2];
			component_plane_arrows[i].geometry.verticesNeedUpdate = true;
		}
	} else {
		for(i = 0; i < component_plane_stars.length; ++i) {
			scene.remove(component_plane_stars[i]);
			scene.remove(component_plane_arrows[i]);
		}
		component_plane_stars = [];
		component_plane_arrows = [];
		for(i = 0; i < new_component_plane_stars.length; ++i) {
			var star_geometry = new THREE.SphereGeometry(0.05, 32, 32);
			var star_material = new THREE.MeshPhongMaterial({color: 0xff0000});
			var star = new THREE.Mesh(star_geometry, star_material);
			star.position.set(new_component_plane_stars[i][0], new_component_plane_stars[i][1], 
				new_component_plane_stars[i][2]);
			component_plane_stars.push(star);
			scene.add(star);

			var arrow_geometry = new THREE.Geometry();
			arrow_geometry.dynamic = true;
			arrow_geometry.vertices.push(new THREE.Vector3(0,0,1));
			arrow_geometry.vertices.push(new THREE.Vector3(new_component_plane_stars[i][0], 
				new_component_plane_stars[i][1], new_component_plane_stars[i][2]));
			arrow_geometry.vertices.push(new THREE.Vector3(new_component_stars[i][0], 
				new_component_stars[i][1], new_component_stars[i][2]));
			var arrow_material = new THREE.LineBasicMaterial({
					color: 0xff0000,
					opacity: 0.9,
					linewidth: 2
			});
			var arrow = new THREE.Line(arrow_geometry, arrow_material);
			arrow.position.set(0,0,0);

			component_plane_arrows.push(arrow);
			scene.add(arrow);
		}
	}

	// Update husimi
	if (new_husimi.length != 0) {
		if (husimi_pts == false) {
			var husimi_geometry = new THREE.Geometry();
			husimi_geometry.dynamic = true;
			var husimi_colors = [];
			for (i = 0; i < new_husimi.length; ++i) {
				var pt = new THREE.Vector3();
				pt.x = new_husimi[i][1][0];
				pt.y = new_husimi[i][1][1];
				pt.z = new_husimi[i][1][2];
				husimi_geometry.vertices.push(pt);

				var color = new THREE.Color();
				color.setHSL(0.6, 0, new_husimi[i][0]);
				husimi_colors.push(color);
			}
			husimi_geometry.colors = husimi_colors;
			var husimi_material = new THREE.PointsMaterial({ vertexColors: THREE.VertexColors, size: 0.1 });
			husimi_material.transparent = true;
			husimi_material.opacity = 1;
			husimi_pts = new THREE.Points(husimi_geometry, husimi_material);
			scene.add(husimi_pts);
		} else {
			for (i = 0; i < new_husimi.length; ++i) {
				husimi_pts.geometry.colors[i].setHSL(0.6, 0, new_husimi[i][0]);
			}
			husimi_pts.geometry.colorsNeedUpdate = true;
		}
	} else {
		if (husimi_pts != false) {
			scene.remove(husimi_pts);
			husimi_pts = false;
		}
	}

	// Update status
	document.getElementById("state").innerHTML = new_state;

	// Update help
	if (dt < 0) {
		document.getElementById("dt").innerHTML = new_dt.toFixed(4);
	} else {
		document.getElementById("dt").innerHTML = new_dt.toFixed(4) + " ";
	}	

	// Update measurements
	if (new_controls == "") {
		control_pane = document.getElementById("controls");
	    control_pane.style.display = "none";
	    document.getElementById("ctrls").innerHTML = "";
	} else {
		control_pane = document.getElementById("controls");
	    control_pane.style.display = "block";
	    if (new_controls != document.getElementById("ctrls").innerHTML) {
	    	document.getElementById("ctrls").innerHTML = new_controls;
	    }
	}	
}

function animate () {
	requestAnimationFrame(animate);
	camera_controls.update();
	renderer.render(scene, camera);
};
animate();