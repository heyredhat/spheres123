import math
import cmath
import gellman
import qutip as qt
import numpy as np
from magic import *

class Sphere:
    def __init__(self, state=None,\
                       energy=None,\
                       dt=0.01,\
                       evolving=False,\
                       show_phase=True,\
                       show_components=False,\
                       show_husimi=False,\
                       show_projection=False,\
                       show_controls=False):
        self.state = state if state != None else qt.rand_ket(2)
        self.energy = energy if energy != None else qt.rand_herm(self.n())

        self.dt = dt
        self.evolving = evolving
        self.show_phase = show_phase
        self.show_components = show_components
        self.show_projection = show_projection
        self.show_husimi = show_husimi
        self.show_controls = show_controls

        self.precalc_bases = None
        self.precalc_paulis = None
        self.precalc_energy_eigs = None
        self.precalc_coherents = None
        self.dim_change = False

    def n(self):
        return self.state.shape[0]

    def random_state(self):
        self.state = qt.rand_ket(self.n())

    def random_energy(self):
        self.energy = qt.rand_herm(self.n())
        self.eigenenergies(reset=True)

    def spin(self):
        return (self.n()-1.)/2.

    def spin_axis2(self):
        mink = np.array([qt.expect(qt.identity(self.n()), self.state),\
                         qt.expect(qt.jmat(self.spin(), "x"), self.state),\
                         -1*qt.expect(qt.jmat(self.spin(), "y"), self.state),\
                         -1*qt.expect(qt.jmat(self.spin(), "z"), self.state)])
        axis = normalize(mink)[1:]
        direction = normalize(axis).tolist()
        length = np.linalg.norm(axis)
        return [direction, length]

    def spin_axis(self):
        direction = np.array([qt.expect(qt.jmat(self.spin(), "x"), self.state),\
                              -1*qt.expect(qt.jmat(self.spin(), "y"), self.state),\
                              -1*qt.expect(qt.jmat(self.spin(), "z"), self.state)])
        spin_squared = np.sqrt(np.sum(direction**2))
        return [normalize(direction).tolist(), spin_squared]

    def phase(self):
        vec = self.state.full().T[0]
        p = np.exp(1j*np.angle(np.sum(vec)))
        return [p.real, p.imag]

    def evolve(self, operator, dt=None, inverse=False):
        if dt == None:
            dt = self.dt
        unitary = (-2*math.pi*1j*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state = unitary*self.state

    def rotate(self, pole, inverse=False):
        if pole == "x":
            self.evolve(qt.jmat(self.spin(), "x"), dt=self.dt, inverse=inverse)
        elif pole == "y":
            self.evolve(qt.jmat(self.spin(), "y"), dt=self.dt, inverse=inverse)
        elif pole == "z":
            self.evolve(qt.jmat(self.spin(), "z"), dt=self.dt, inverse=inverse)

    def rotate_star(self, index, pole, dt=0.01, inverse=False):
        #print("*****")
        #print(self.state)
        roots = q_SurfaceXYZ(self.state)
        #print(roots)
        root = roots[index]
        #print(root)
        root_state = SurfaceXYZ_q([root])
        #print(root_state)
        root_state = evolver(root_state, qt.jmat(0.5, pole), dt=self.dt, inverse=inverse)
       #print(root_state)
        new_xyz = q_SurfaceXYZ(root_state)[0]
        #print(new_xyz)
        roots[index] = new_xyz
        #print(roots)
        #print(SurfaceXYZ_q(roots))
        #print("******")
        self.state = SurfaceXYZ_q(roots)

    def rotate_component(self, index, pole, dt=0.01, inverse=False, unitary=True):
        polynomial = v_polynomial(self.state.full().T[0])
        component = polynomial[index]
        component_xyz = c_xyz(component)
        component_state = SurfaceXYZ_q([component_xyz])
        component_state = evolver(component_state, qt.jmat(0.5, pole), dt=self.dt, inverse=inverse)
        new_xyz = q_SurfaceXYZ(component_state)[0]
        new_component = xyz_c(new_xyz)
        if (new_component != float('Inf')):
            polynomial[index] = new_component
        new_vector = polynomial_v(polynomial)
        if unitary:
            self.state = qt.Qobj(new_vector).unit()
        else:
            self.state = qt.Qobj(new_vector)

    def collapse(self, operator):
        L, V = operator.eigenstates()
        amplitudes = [self.state.overlap(v) for v in V]
        probabilities = np.array([(a*np.conjugate(a)).real for a in amplitudes])
        probabilities = probabilities/probabilities.sum()
        pick = np.random.choice(list(range(len(V))), 1, p=probabilities)[0]
        projector = V[pick].ptrace(0)
        self.state = (projector*self.state).unit()
        #print("collapsed!")
        return pick, L, probabilities

    def update(self):
        if self.evolving and self.energy != None:
          self.evolve(self.energy)

    def stars(self):
        return q_SurfaceXYZ(self.state)

    def plane_stars(self):
        C = v_C(self.state.full().T[0])
        xyz = []
        for c in C:
            if c == float('Inf'):
                xyz.append([1000,1000,0])
            else:
                xyz.append([c.real, c.imag, 0])
        return xyz
 
    def component_stars(self):
        #components = self.state.full().T[0].tolist()
        #return [c_xyz(c) for c in components]
        return [c_xyz(c) for c in v_polynomial(self.state.full().T[0])]

    def plane_component_stars(self):
       #C = self.state.full().T[0].tolist()
       #return [[c.real, c.imag, 0] for c in C]
       C = v_polynomial(self.state.full().T[0])
       return [[c.real, c.imag, 0] for c in C] 

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
        return stuff

    def set_stars(self, new_stars):
        self.state = SurfaceXYZ_q(new_stars)

    def create_star(self):
        xyz = q_SurfaceXYZ(qt.rand_ket(2))
        self.state = SurfaceXYZ_q(self.stars() + xyz).unit()
        self.energy = qt.rand_herm(self.n())
        self.hermitian_bases(reset=True)
        self.paulis(reset=True)
        self.eigenenergies(reset=True)
        self.dim_change = True

    def destroy_star(self):
        if self.n() > 2:
            self.state = SurfaceXYZ_q(self.stars()[1:]).unit()
            self.energy = qt.rand_herm(self.n())
            self.hermitian_bases(reset=True)
            self.paulis(reset=True)
            self.eigenenergies(reset=True)
            self.dim_change = True

    def pretty_state(self):
        vec = self.state.full().T[0]
        s = np.array_str(vec, max_line_width=500, precision=2, suppress_small=True) + " aka<br />"
        s += "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[" + " ".join(["%.2f^%.2f" % (abs(c), cmath.phase(c)) for c in self.state.full().T[0]]) + "]"
        return s

    def husimi2(self):
        N = 25
        theta = np.linspace(0, math.pi, N)
        phi = np.linspace(0, 2*math.pi, N)
        Q, THETA, PHI = qt.spin_q_function(self.state, theta, phi)
        pts = []
        for i, j, k in zip(Q, THETA, PHI):
            for q, t, p in zip(i, j, k):
                pts.append([q, sph_xyz(t, p)])
        return pts

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

    def hermitian_bases(self, reset=False):
        if self.precalc_bases == None or reset == True:
            self.precalc_bases = [[qt.Qobj(basis), qt.Qobj(basis).eigenstates()] for basis in gellman.get_basis(self.n())]
        return self.precalc_bases

    def hermitian_basis(self):
        bases = self.hermitian_bases()
        vector = [qt.expect(basis[0], self.state) for basis in bases]
        return vector, bases

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

    def controls(self):
        s = ""
        ops, eigs = self.paulis()
        signs = ["X", "Y", "Z"]
        for i in range(len(ops)):
            op = ops[i]
            L, V = eigs[i]
            s += "     %s '%d': %.2f\n" % (signs[i], i+1, qt.expect(op, self.state))
            for j in range(len(V)):
                amplitude = self.state.overlap(V[j])
                probability = (amplitude*np.conjugate(amplitude)).real
                s += "\t%.2f\t%.2f%%\n" % (L[j], probability*100)
        s += "     H '4': %.2f (energy)\n" % (qt.expect(self.energy, self.state))
        L, V = self.eigenenergies()
        for j in range(len(V)):
            amplitude = self.state.overlap(V[j])
            probability = (amplitude*np.conjugate(amplitude)).real
            s += "\t%.2f\t%.2f%%\n" % (L[j], probability*100)
        return s[:-1]

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
                s += "\t%.2f : %s | %.2f\n" % (L[j].real,'({0.real:.2f} + {0.imag:.2f}i)'.format(inner), (inner*np.conjugate(inner)).real)
        return s[:-1]