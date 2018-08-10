import sys
import math
import cmath
import gellman
import qutip as qt
import numpy as np
from magic import *
from spheres import *

class PureSphere:
    def __init__(self, state=None,\
                       energy=None,\
                       dt=0.01,\
                       evolving=False,
                       evolution="spin",\
                       double=None):
        self.state = state if state != None else qt.rand_ket(2)
        self.energy = energy if energy != None else qt.rand_herm(self.n())
        self.dt = dt
        self.evolving = evolving
        self.evolution = evolution
        self.dimensionality = None
        self.double = double

        self.precalc_stars = None
        self.precalc_plane_stars = None
        self.precalc_component_stars = None
        self.precalc_plane_component_stars = None
        self.precalc_polynomial = None
        self.precalc_qubits = None
        self.precalc_symmetrical = None
        self.precalc_bases = None
        self.precalc_paulis = None
        self.precalc_energy_eigs = None
        self.precalc_coherents = None
        self.precalc_1Dfock = None
        self.precalc_1Dops = None
        self.precalc_2Dfock = None
        self.precalc_2Dops = None
        self.precalc_1Dstuff = None
        self.precalc_2Dstuff = None
        self.dim_change = False

##################################################################################################################

    def n(self):
        return self.state.shape[0]

    def spin(self):
        return (self.n()-1.)/2.

    def spin_axis(self):
        direction = np.array([qt.expect(self.paulis()[0][0], self.state).real,\
                              -1*qt.expect(self.paulis()[0][1], self.state).real,\
                              -1*qt.expect(self.paulis()[0][2], self.state).real])
        spin_squared = np.sqrt(np.sum(direction**2))
        return [normalize(direction).tolist(), spin_squared]

    def phase(self):
        vec = self.state.full().T[0]
        p = np.exp(1j*np.angle(np.sum(vec)))
        return [p.real, p.imag]

    def paulis(self, reset=False):
        if self.precalc_paulis == None or reset == True:
            ops = qt.jmat(self.spin())
            eigs = [op.eigenstates() for op in ops]
            self.precalc_paulis = [ops, eigs]
        return self.precalc_paulis

    def eigenenergies(self, reset=False):
        if self.precalc_energy_eigs == None and self.energy != None or reset == True:
            self.precalc_energy_eigs = self.energy.eigenstates()
        return self.precalc_energy_eigs

    def hermitian_bases(self, reset=False):
        if self.precalc_bases == None or reset == True:
            self.precalc_bases = [[qt.Qobj(basis), qt.Qobj(basis).eigenstates()]\
                for basis in gellman.get_basis(self.n())]
        return self.precalc_bases

    def hermitian_basis(self):
        bases = self.hermitian_bases()
        vector = [qt.expect(basis[0], self.state) for basis in bases]
        return vector, bases
   
    def coherent_states(self, N=25, reset=False):
        if self.precalc_coherents == None or reset == True or self.dim_change == True:
            theta = np.linspace(0, math.pi, N)
            phi = np.linspace(0, 2*math.pi, N)
            THETA, PHI = np.meshgrid(theta, phi)
            self.precalc_coherents = [[qt.spin_coherent(self.spin(), THETA[i][j], PHI[i][j])\
                            for j in range(N)] for i in range(N)], THETA, PHI
            self.dim_change = False
        return self.precalc_coherents 

    def husimi(self):
        N = 25
        coherents, THETA, PHI = self.coherent_states(N=N)
        Q = np.zeros_like(THETA)
        for i in range(N):
            for j in range(N):
                amplitude = self.state.overlap(coherents[i][j])
                probability = (amplitude*np.conjugate(amplitude)).real
                Q[-1*i][-1*j] = probability
        pts = []
        for i, j, k in zip(Q, THETA, PHI):
            for q, t, p in zip(i, j, k):
                pts.append([q, sph_xyz(t, p)])
        return pts

    def husimi_old(self):
        N = 25
        theta = np.linspace(0, math.pi, N)
        phi = np.linspace(0, 2*math.pi, N)
        Q, THETA, PHI = qt.spin_q_function(self.state, theta, phi)
        pts = []
        for i, j, k in zip(Q, THETA, PHI):
            for q, t, p in zip(i, j, k):
                pts.append([q, sph_xyz(t, p)])
        return pts

    def stars(self, reset=False):
        if self.precalc_stars == None or reset == True:
            self.precalc_stars = q_SurfaceXYZ(self.state)
        return self.precalc_stars

    def plane_stars(self, reset=False):
        if self.precalc_plane_stars == None or reset == True:
            C = v_C(self.state.full().T[0])
            xyz = []
            for c in C:
                if c == float('Inf'):
                    xyz.append([1000,1000,0])
                else:
                    xyz.append([c.real, c.imag, 0])
            self.precalc_plane_stars = xyz
        return self.precalc_plane_stars
            
    def component_stars(self, reset=False):
        if self.precalc_component_stars == None or reset == True:
            self.precalc_component_stars = [c_xyz(c) for c in self.polynomial()]
        return self.precalc_component_stars

    def plane_component_stars(self, reset=False):
        if self.precalc_plane_component_stars == None or reset == True:
            C = self.polynomial()
            self.precalc_plane_component_stars = [[c.real, c.imag, 0] for c in C] 
        return self.precalc_plane_component_stars 

    def polynomial(self, reset=False):
        if self.precalc_polynomial == None or reset == True:
            self.precalc_polynomial = v_polynomial(self.state.full().T[0])
        return self.precalc_polynomial

    def qubits(self, reset=False):
        if self.precalc_qubits == None or reset == True:
            self.precalc_qubits = [SurfaceXYZ_q([xyz]) for xyz in self.stars()]
        return self.precalc_qubits

    def symmetrical(self, reset=False):
        if self.precalc_symmetrical == None or reset == True:
            self.precalc_symmetrical = symmeterize(self.qubits())
        return self.precalc_symmetrical

    def allstars(self, stars=True, plane_stars=True, component_stars=True, plane_component_stars=True):
        stuff = {}
        polynomial = v_polynomial(self.state.full().T[0])
        stuff["component_stars"] = [c_xyz(c) for c in polynomial] if component_stars else []
        stuff["plane_component_stars"] = [[c.real, c.imag, 0] for c in polynomial] if plane_component_stars else []
        if stars or plane_stars:
            roots = polynomial_C(polynomial) 
            if plane_stars:
                planeStars = []
                for c in roots:
                    if c == float('Inf'):
                        planeStars.append([1000,1000,0])
                    else:
                        planeStars.append([c.real, c.imag, 0])
                stuff["plane_stars"] = planeStars
            else:
                stuff["plane_stars"] = []
            stuff["stars"] = [c_xyz(c) for c in roots] if stars else []
        self.precalc_stars = stuff["stars"] 
        self.precalc_plane_stars = stuff["plane_stars"]
        self.precalc_component_stars = stuff["component_stars"]
        self.precalc_plane_component_stars = stuff["plane_component_stars"]  
        self.precalc_polynomial = polynomial  
        self.precalc_qubits = None     
        self.precalc_symmetrical = None       
        return stuff

    def sym_arrows(self):
        symmeterized = self.symmetrical().copy()
        symmeterized.dims = [[2]*(self.n()-1),[1]*(self.n()-1)]
        sym_bits = [symmeterized.ptrace(i) for i in range(self.n()-1)]
        return [xyz_radial([qt.expect(qt.sigmax(), bit),\
                            -1*qt.expect(qt.sigmay(), bit),\
                            -1*qt.expect(qt.sigmaz(), bit)]) for bit in sym_bits]

    def distinguishable_pieces(self):
        if self.dimensionality != None:
            state_copy = self.state.copy()
            state_copy.dims = [self.dimensionality, [1]*len(self.dimensionality)]
            pieces = [state_copy.ptrace(i) for i in range(len(self.dimensionality))]
            return pieces
        else:
            return [self.state.ptrace(0)]

    def dist_pieces_spin(self, pieces):
        arrows = []
        for piece in pieces:
            j = dim_spin(piece.shape[0]) 
            direction = np.array([qt.expect(qt.jmat(j, "x"), piece).real,\
                                  -1*qt.expect(qt.jmat(j, "y"), piece).real,\
                                  -1*qt.expect(qt.jmat(j, "z"), piece).real])
            spin_squared = np.sqrt(np.sum(direction**2))
            arrows.append([normalize(direction).tolist(), spin_squared])
        return arrows

    def are_separable(self, pieces):
        seps = []
        for piece in pieces:
            entropy = qt.entropy_vn(piece) 
            if entropy < 0.001 and entropy > -0.001:
                seps.append(True)
            else:
                seps.append(False)
        return seps

    def separable_skies(self, pieces, are_separable):
        skies = {}
        for i in range(len(pieces)):
            if are_separable[i] == True:
                q = density_to_purevec(pieces[i])
                skies[i] = q_SurfaceXYZ(q)
        return skies

##################################################################################################################

    def refresh(self, dimchange=None):
        ##print("P")
        ##print(self.state)
        ##print("Q")
        self.double.signal_pure_update()
        if dimchange:
            self.precalc_bases = None
            self.precalc_paulis = None
            self.precalc_energy_eigs = None
            self.precalc_coherents = None
            self.precalc_1Dops = None
            self.precalc_2Dops = None
        self.precalc_stars = None
        self.precalc_plane_stars = None
        self.precalc_component_stars = None
        self.precalc_plane_component_stars = None
        self.precalc_polynomial = None
        self.precalc_qubits = None
        self.precalc_symmetrical = None
        self.precalc_1Dfock = None
        self.precalc_2Dfock = None
        self.precalc_1Dstuff = None
        self.precalc_2Dstuff = None

    def create_star(self):
        xyz = q_SurfaceXYZ(qt.rand_ket(2))
        self.state = SurfaceXYZ_q(self.stars() + xyz).unit()
        self.energy = qt.rand_herm(self.n())
        self.refresh(dimchange=True)
        self.dimensionality = None
        return self.state

    def destroy_star(self):
        if self.n() > 2:
            self.state = SurfaceXYZ_q(self.stars()[1:]).unit()
            self.energy = qt.rand_herm(self.n())
            self.refresh(dimchange=True)
            self.dimensionality = None
            return self.state

    def random_state(self):
        self.state = qt.rand_ket(self.n())
        self.refresh()
        return self.state

    def random_energy(self):
        self.energy = qt.rand_herm(self.n())
        self.eigenenergies(reset=True)
    
    def evolve(self, operator, dt=None, inverse=False):
        if dt == None:
            dt = self.dt
        unitary = (-2*math.pi*1j*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state = unitary*self.state
        return self.state

##################################################################################################################

    def rotate(self, pole, dt=None, inverse=False):
        if dt == None:
            dt = self.dt
        if pole == "x":
            self.evolve(self.paulis()[0][0], dt=dt, inverse=inverse)
        elif pole == "y":
            self.evolve(self.paulis()[0][1], dt=dt, inverse=inverse)
        elif pole == "z":
            self.evolve(self.paulis()[0][2], dt=dt, inverse=inverse)
        self.refresh()
        return self.state

    def rotate_star(self, index, pole, dt=0.01, inverse=False):
        roots = self.stars()
        root = roots[index]
        root_state = SurfaceXYZ_q([root])
        root_state = evolver(root_state, qt.jmat(0.5, pole), dt=self.dt, inverse=inverse)
        new_xyz = q_SurfaceXYZ(root_state)[0]
        roots[index] = new_xyz
        self.state = SurfaceXYZ_q(roots)
        self.refresh()
        return self.state

    def rotate_component(self, index, pole, dt=0.01, inverse=False, unitary=True):
        polynomial = self.polynomial()
        component = polynomial[index]
        component_xyz = c_xyz(component)
        component_state = SurfaceXYZ_q([component_xyz])
        component_state = evolver(component_state, qt.jmat(0.5, pole), dt=self.dt, inverse=inverse)
        new_xyz = q_SurfaceXYZ(component_state)[0]
        new_component = xyz_c(new_xyz)
        if (new_component != float('Inf')):
            polynomial[index] = new_component
        else:
            return self.state
        new_vector = polynomial_v(polynomial)
        #if unitary:
        self.state = qt.Qobj(new_vector).unit()
        #else:
        #    self.state = qt.Qobj(new_vector)
        #print("(")
        #print(self.state)
        #print(")")
        self.refresh()
        return self.state

    def rotate_symmetrical(self, direction, i, dt, inverse=False):
        sym = self.symmetrical()
        op = None
        if direction == "x":
            op = 0.5*qt.sigmax()
        elif direction == "y":
            op = 0.5*qt.sigmay()
        elif direction == "z":
            op = 0.5*qt.sigmaz()
        total_op = op if i == 0 else qt.identity(2)
        for j in range(1, self.n()-1):
            if j == i:
                total_op = qt.tensor(total_op, op)
            else:
                total_op = qt.tensor(total_op, qt.identity(2))
        total_op.dims = [[(2**(self.n()-1))], [2**(self.n()-1)]]
        sym2 = evolver(sym, total_op, dt=dt, inverse=inverse)
        self.state = unsymmeterize(sym2)
        self.refresh()
        return self.state

    def rotate_distinguishable(self, i, direction, dt=0.01, inverse=False):
        if self.dimensionality != None:
            j = dim_spin(self.dimensionality[i])
            op = qt.jmat(j, direction)
            total_op = op if i == 0 else qt.identity(self.dimensionality[0])
            for j in range(1, len(self.dimensionality)):
                if j == i:
                    total_op = qt.tensor(total_op, op)
                else:
                    total_op = qt.tensor(total_op, qt.identity(self.dimensionality[j]))
            total_op.dims = [[self.n()], [self.n()]]
            self.state = evolver(self.state, total_op, dt=dt, inverse=inverse)
            self.refresh()
            return self.state

    def mink_rotate(self, direction, angle, inverse=False):
        self.state = mink_rotate_qubits(self.qubits(), direction, dt=angle, inverse=inverse)
        self.refresh()
        return self.state

    def boost(self, direction, rapidity, inverse=False):
        if inverse == False:
            inverse = True
        else:
            inverse = False
        self.state = mink_boost_qubits(self.qubits(), direction, dt=rapidity, inverse=inverse)
        self.refresh()
        return self.state

##################################################################################################################

    def collapse(self, operator):
        L, V = operator.eigenstates()
        amplitudes = [self.state.overlap(v) for v in V]
        probabilities = np.array([(a*np.conjugate(a)).real for a in amplitudes])
        probabilities = probabilities/probabilities.sum()
        pick = np.random.choice(list(range(len(V))), 1, p=probabilities)[0]
        try:
            vec = V[pick].full().T[0]
            projector = qt.Qobj(np.outer(vec,np.conjugate(vec)))
            self.state = (projector*self.state).unit()
            self.refresh()
            return L[pick], L, probabilities
        except Exception as e:
            #print("collapse error!: %s " % e)
            sys.stdout.flush() 

    def symmetrical_collapse(self, direction, i):
        sym = self.symmetrical()
        op = None
        if direction == "x":
            op = 0.5*qt.sigmax()
        elif direction == "y":
            op = 0.5*qt.sigmay()
        elif direction == "z":
            op = 0.5*qt.sigmaz()
        elif direction == "h":
            return None
        elif direction == "r":
            op = qt.rand_herm(2)
        total_op = op if i == 0 else qt.identity(2)
        for j in range(1, self.n()-1):
            if j == i:
                total_op = qt.tensor(total_op, op)
            else:
                total_op = qt.tensor(total_op, qt.identity(2))
        total_op.dims = [[(2**(self.n()-1))], [2**(self.n()-1)]]
        L, V = total_op.eigenstates()
        amplitudes = [sym.overlap(v) for v in V]
        probabilities = np.array([(a*np.conjugate(a)).real for a in amplitudes])
        probabilities = probabilities/probabilities.sum()
        pick = np.random.choice(list(range(len(V))), 1, p=probabilities)[0]
        vec = V[pick].full().T[0]
        projector = qt.Qobj(np.outer(vec,np.conjugate(vec)))
        sym = (projector*sym).unit()
        self.state = unsymmeterize(sym)
        self.refresh()
        return L[pick], L, probabilities

    def distinguishable_collapse(self, i, direction):
        if self.dimensionality != None:
            j = dim_spin(self.dimensionality[i])
            op = None
            if direction == "x" or direction == "y" or direction == "z":
                op = qt.jmat(j, direction)
            elif direction == "r":
                op = qt.rand_herm(self.dimensionality[i])
            total_op = op if i == 0 else qt.identity(self.dimensionality[0])
            for j in range(1, len(self.dimensionality)):
                if j == i:
                    total_op = qt.tensor(total_op, op)
                else:
                    total_op = qt.tensor(total_op, qt.identity(self.dimensionality[j]))
            total_op.dims = [[self.n()], [self.n()]]
            L, V = total_op.eigenstates()
            amplitudes = [self.state.overlap(v) for v in V]
            probabilities = np.array([(a*np.conjugate(a)).real for a in amplitudes])
            probabilities = probabilities/probabilities.sum()
            pick = np.random.choice(list(range(len(V))), 1, p=probabilities)[0]
            vec = V[pick].full().T[0]
            projector = qt.Qobj(np.outer(vec,np.conjugate(vec)))
            self.state = (projector*self.state).unit()
            self.refresh()
            return L[pick], L, probabilities

##################################################################################################################

    def harmonic_oscillator_1D(self):
        #fock = symmeterize(q_qubits(self.state))
        fock = self.state.copy()
        self.precalc_1Dfock = fock
        ops = construct_1dfock_operators(fock.shape[0])
        self.precalc_1Dops = ops
        stuff = {}
        stuff["position"] = xyz_radial([qt.expect(ops["position"], fock), 0, 0])
        stuff["momentum"] = xyz_radial([qt.expect(ops["momentum"], fock), 0, 0])
        stuff["number"] = qt.expect(ops["number"], fock)
        stuff["energy"] = qt.expect(ops["energy"], fock)
        self.precalc_1Dstuff = stuff
        return stuff

    def fock1D(self):
        if self.precalc_1Dstuff == None:
            self.precalc1Dstuff = self.harmonic_oscillator_1D()
        stuff = self.precalc_1Dstuff
        if self.precalc_1Dfock == None:
            #self.precalc_1Dfock = self.symmetrical()
            self.precalc_1Dfock = self.state.copy()
        fock = self.precalc_1Dfock
        if self.precalc_1Dops == None:
            self.precalc_1Dops = construct_1dfock_operators(fock.shape[0])
        ops = self.precalc_1Dops
        return (fock, ops, stuff)

    def harmonic_oscillator_1D_evolve(self, dt=0.01):
        if self.precalc_1Dfock == None:
           # self.precalc_1Dfock = self.symmetrical()
           self.precalc_1Dfock = self.state.copy()
        fock = self.precalc_1Dfock
        if self.precalc_1Dops == None:
            self.precalc_1Dops = construct_1dfock_operators(fock.shape[0])
        ops = self.precalc_1Dops
        unitary = (-2*math.pi*1j*ops["energy"]*self.dt*0.1).expm()
        fock = unitary*fock
        #self.state = unsymmeterize(fock)
        self.state = fock
        self.refresh()
        return self.state

    def harmonic_oscillator_1D_collapse(self, kind):
        if self.precalc_1Dfock == None:
            #self.precalc_1Dfock = self.symmetrical()
            self.precalc_1Dfock = self.state.copy()
        fock = self.precalc_1Dfock
        if self.precalc_1Dops == None:
            self.precalc_1Dops = construct_1dfock_operators(fock.shape[0])
        ops = self.precalc_1Dops
        L, V = ops[kind].eigenstates()
        amplitudes = [fock.overlap(v) for v in V]
        probabilities = np.array([(a*np.conjugate(a)).real for a in amplitudes])
        probabilities = probabilities/probabilities.sum()
        pick = np.random.choice(list(range(len(V))), 1, p=probabilities)[0]
        vec = V[pick].full().T[0]
        projector = qt.Qobj(np.outer(vec,np.conjugate(vec)))
        #self.state = unsymmeterize((projector*fock).unit())
        self.state = (projector*fock).unit()
        self.refresh()
        return L[pick], L, probabilities

    def harmonic_oscillator_2D(self):
        fock = spin_fock(self.state)
        self.precalc_2Dfock = fock
        ops = construct_2dfock_operators(self.n())
        self.precalc_2Dops = ops
        stuff = {}
        stuff["position"] = xyz_radial([qt.expect(ops["X"]["position"], fock),\
                                        qt.expect(ops["Y"]["position"], fock),\
                                        0])
        stuff["momentum"] = xyz_radial([qt.expect(ops["X"]["momentum"], fock),\
                                        qt.expect(ops["Y"]["momentum"], fock),\
                                        0])
        stuff["angmo"] = xyz_radial([qt.expect(ops["Jx"], fock),\
                                     qt.expect(ops["Jy"], fock),\
                                     qt.expect(ops["Jz"], fock)])
        stuff["number"] = [qt.expect(ops["X"]["number"], fock),\
                           qt.expect(ops["Y"]["number"], fock)]
        stuff["energy"] = qt.expect(ops["T"], fock)
        self.precalc_2Dstuff = stuff
        return stuff

    def fock2D(self):
        if self.precalc_2Dstuff == None:
            self.precalc2Dstuff = self.harmonic_oscillator_2D()
        stuff = self.precalc_2Dstuff
        if self.precalc_2Dfock == None:
            self.precalc_2Dfock = spin_fock(self.state)
        fock = self.precalc_2Dfock
        if self.precalc_2Dops == None:
            self.precalc_2Dops = construct_2dfock_operators(self.n())
        ops = self.precalc_2Dops
        return (fock, ops, stuff)

    def harmonic_oscillator_2D_evolve(self, dt=0.01):
        if self.precalc_2Dfock == None:
            self.precalc_2Dfock = spin_fock(self.state)
        fock = self.precalc_2Dfock
        if self.precalc_2Dops == None:
            self.precalc_2Dops = construct_2dfock_operators(self.n())
        ops = self.precalc_2Dops
        unitary = (-2*math.pi*1j*ops["T"]*self.dt*0.1).expm()
        fock = unitary*fock
        fock.dims = [[fock.shape[0]], [1]]
        self.state = fock_spin(fock)
        self.refresh()
        return self.state

    def harmonic_oscillator_2D_collapse(self, kind):
        if self.precalc_2Dfock == None:
            self.precalc_2Dfock = spin_fock(self.state)
        fock = self.precalc_2Dfock
        if self.precalc_2Dops == None:
            self.precalc_2Dops = construct_2dfock_operators(self.n())
        ops = self.precalc_2Dops
        L, V = ops[kind].eigenstates()
        amplitudes = [fock.overlap(v) for v in V]
        probabilities = np.array([(a*np.conjugate(a)).real for a in amplitudes])
        probabilities = probabilities/probabilities.sum()
        pick = np.random.choice(list(range(len(V))), 1, p=probabilities)[0]
        vec = V[pick].full().T[0]
        projector = qt.Qobj(np.outer(vec,np.conjugate(vec)))
        self.state = fock_spin((projector*fock).unit())
        self.refresh()
        return L[pick], L, probabilities

##################################################################################################################

    def pretty_state(self):
        vec = self.state.full().T[0]
        s = np.array_str(vec, max_line_width=500, precision=2, suppress_small=True) + " aka<br />"
        s += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[" + " ".join(["%.2f^%.2f" %\
            (abs(c), cmath.phase(c)) for c in self.state.full().T[0]]) + "]"
        return s

    def pretty_measurements(self, harmonic1D=False, harmonic2D=False):
        s = ""
        ops, eigs = self.paulis()
        signs = ["X", "Y", "Z"]
        keys = ["f", "g", "h"]
        for i in range(len(ops)):
            op = ops[i]
            L, V = eigs[i]
            s += "     %s '%s': %.2f\n" % (signs[i], keys[i], qt.expect(op, self.state))
            for j in range(len(V)):
                amplitude = self.state.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "\t%.2f\t%.2f%%\n" % (L[j], probability*100)
        s += "     H 'y': %.2f (energy)\n" % (qt.expect(self.energy, self.state))
        L, V = self.eigenenergies()
        for j in range(len(V)):
            amplitude = self.state.overlap(V[j])
            probability = (amplitude*np.conjugate(amplitude)).real
            s += "\t%.2f\t%.2f%%\n" % (L[j], probability*100)
        s += "     R 't' (random)\n"
        if harmonic1D:
            fock, ops, stuff = self.fock1D()
            s += "   -------------------------------\n"
            s += "\n     1d harmonic oscillator:\n"
            s += "      position ';':\n"
            L, V = ops["position"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      momentum ''':\n"
            L, V = ops["momentum"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      number '\\': %.2f\n" % stuff["number"]
            L, V = ops["number"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      energy '=': %.2f\n" % stuff["energy"]
            L, V = ops["energy"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
        if harmonic2D:
            s += "   -------------------------------\n"
            fock, ops, stuff = self.fock2D()
            s += "\n     2d harmonic oscillator:\n"
            s += "      X position:\n"
            L, V = ops["X"]["position"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      Y position:\n"
            L, V = ops["Y"]["position"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      X momentum:\n"
            L, V = ops["X"]["momentum"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      Y momentum:\n"
            L, V = ops["Y"]["momentum"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      number in x: %.2f\n" % stuff["number"][0]
            L, V = ops["X"]["number"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      number in y: %.2f\n" % stuff["number"][1]
            L, V = ops["Y"]["number"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "      energy: %.2f\n" % stuff["energy"]
            L, V = ops["T"].eigenstates()
            for j in range(len(V)):
                amplitude = fock.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "     \t%.2f\t%.2f%%\n" % (L[j], probability*100)
        return s

    def pretty_hermitian_basis(self):
        vector, bases = self.hermitian_basis()
        s = ""
        for i in range(len(bases)):
            s += "  %d: %.3f\n" % (i, vector[i])
            basis = bases[i][0]
            L, V = bases[i][1]
            for j in range(len(V)):
                l = L[j]
                v = V[j]
                inner = self.state.overlap(v)
                s += "\t%.2f : %s | %.2f\n" % (L[j].real,'({0.real:.2f} + {0.imag:.2f}i)'.format(inner),\
                    (inner*np.conjugate(inner)).real)
        return s[:-1]