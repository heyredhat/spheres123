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
sphere_material.opacity = 0.7;
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

/************************************************************************************************************/

var raycaster = new THREE.Raycaster();
var mouse = new THREE.Vector2();

document.addEventListener( 'mousedown', 
	function ( event ) {
		event.preventDefault();
		mouse.x = ( event.clientX / renderer.domElement.clientWidth ) * 2 - 1;
		mouse.y = - ( event.clientY / renderer.domElement.clientHeight ) * 2 + 1;
		raycaster.setFromCamera( mouse, camera );
		var intersects = raycaster.intersectObjects( stars );
		if ( intersects.length > 0 ) {

			intersects[ 0 ].object.material.color.setHex( Math.random() * 0xffffff );
		}
		}, false );

/************************************************************************************************************/

function start_server () {
	$.ajax({
		url: "/start/",
		dataType: "json",
		success: function (response) { },
		error: function (response) {
			console.log("error!: " + response.responseText);
		},
		always: function (response) {
			console.log("haylp!: " + response.responseText);
		}
	});
}

function stop_server () {
	$.ajax({
		url: "/stop/",
		dataType: "json",
		success: function (response) { },
		error: function (response) {
			console.log("error!: " + response.responseText);
		},
		always: function (response) {
			console.log("haylp!: " + response.responseText);
		}
	});
}

document.addEventListener("keypress", function (event) {
	var keyCode = event.which;
	if (keyCode == 47) {
		help_pane = document.getElementById("help");
    	help_pane.style.display = help_pane.style.display == "none" ? "block" : "none";
    } else if (keyCode == 96) {
    	status_pane = document.getElementById("status");
    	status_pane.style.display = status_pane.style.display == "none" ? "block" : "none";
    } else {
		$.ajax({
			url: "/keypress/?keyCode=" + keyCode,
			dataType: "json",
			success: function (response) { 
				if(response["collapsed"] == true) {
					pick = response["pick"];
					alert(pick);
					start_server();
				}
			},
			error: function (response) {
				console.log("error!: " + response.responseText);
			},
			always: function (response) {
				console.log("haylp!: " + response.responseText);
			}
		});
	}
});

/************************************************************************************************************/

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

	// Update spin axis arrow
	spin_axis_arrow.setDirection(new THREE.Vector3(new_spin_axis[0][0], new_spin_axis[0][1], new_spin_axis[0][2]));
	spin_axis_arrow.setLength(new_spin_axis[1]);

	// Update phase arrow
	if (new_phase.length == 0) {
		phase_arrow.visible = false;
	} else {
		phase_arrow.visible = true;
		phase_arrow.setDirection(new THREE.Vector3(new_phase[0], new_phase[1], 0));
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
		stars = []
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
			component_stars[i].position.set(new_component_stars[i][0], new_component_stars[i][1], new_component_stars[i][2]);
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
			star.position.set(new_component_stars[i][0], new_component_stars[i][1], new_component_stars[i][2]);
			component_stars.push(star);
			scene.add(star);
		}
	}

	if (new_plane_stars.length != 0 || new_component_plane_stars.length != 0) {
		XY_plane.visible = true;
	} else {
		XY_plane.visible = false;
	}

	// Update plane stars
	if (new_plane_stars.length == plane_stars.length) {
		for(i = 0; i < plane_stars.length; ++i) {
			plane_stars[i].position.set(new_plane_stars[i][0], new_plane_stars[i][1], new_plane_stars[i][2]);

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
			arrow_geometry.vertices.push(new THREE.Vector3(new_plane_stars[i][0], new_plane_stars[i][1], new_plane_stars[i][2]));
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
			component_plane_stars[i].position.set(new_component_plane_stars[i][0], new_component_plane_stars[i][1], new_component_plane_stars[i][2]);

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
			star.position.set(new_component_plane_stars[i][0], new_component_plane_stars[i][1], new_component_plane_stars[i][2]);
			component_plane_stars.push(star);
			scene.add(star);

			var arrow_geometry = new THREE.Geometry();
			arrow_geometry.dynamic = true;
			arrow_geometry.vertices.push(new THREE.Vector3(0,0,1));
			arrow_geometry.vertices.push(new THREE.Vector3(new_component_plane_stars[i][0], new_component_plane_stars[i][1], new_component_plane_stars[i][2]));
			arrow_geometry.vertices.push(new THREE.Vector3(new_component_stars[i][0], new_component_stars[i][1], new_component_stars[i][2]));
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
			stop_server();

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

			start_server();
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
		document.getElementById("dt").innerHTML = new_dt.toFixed(3);
	} else {
		document.getElementById("dt").innerHTML = new_dt.toFixed(3) + " ";
	}	

	// Update controls
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

var spheresSocket = io.connect(null, {port: location.port, rememberTransport: false});
spheresSocket.on("animate", function(socketData) {
	render(JSON.parse(socketData));
});

function animate () {
	requestAnimationFrame(animate);
	camera_controls.update();
	renderer.render(scene, camera);
};
animate();