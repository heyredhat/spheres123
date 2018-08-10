import sys
import math
import cmath
import gellman
import qutip as qt
import numpy as np
from magic import *
from spheres import *
from puresphere import *

class MixedSphere:
    def __init__(self, parent=None,\
                       energy=None,\
                       dt=0.01,\
                       evolving=False):
        self.parent = parent
        self.state = None
        self.energy = energy 
        self.dt = dt
        self.evolving = evolving
        self.dimensionality = None
        self.evolution = "spin"

        self.precalc_bases = None
        self.precalc_paulis = None
        self.precalc_energy_eigs = None
        self.precalc_coherents = None

        self.pure_sphere = None

##################################################################################################################

    def n(self):
        if self.pure_sphere == None:
            if self.state == None:
                self.state = self.parent.child_state(self)
            return self.state.shape[0]
        else:
            return self.pure_sphere.n()

    def spin(self):
        if self.pure_sphere == None:
            return (self.n()-1.)/2.
        else:
            return self.pure_sphere.spin()

    def spin_axis(self):
        #print("******")
        #print(self.state)
        #if self.pure_sphere != None:
            #print(self.pure_sphere.state)
        #print(self.parent.children)
        #print(self.parent.state)
        #print(self.parent.dims)
        #print("*******")

        if self.pure_sphere == None:
            direction = np.array([qt.expect(self.paulis()[0][0], self.state).real,\
                                  -1*qt.expect(self.paulis()[0][1], self.state).real,\
                                  -1*qt.expect(self.paulis()[0][2], self.state).real])
            spin_squared = np.sqrt(np.sum(direction**2))
            return [normalize(direction).tolist(), spin_squared]
        else:
            return self.pure_sphere.spin_axis()

    def phase(self):
        if self.pure_sphere == None:
            mat = self.state.full()
            p = np.exp(1j*np.angle(np.sum(mat)))
            return [p.real, p.imag]
        else:
            return self.pure_sphere.phase()

    def paulis(self, reset=False):
        if self.pure_sphere == None:
            if self.precalc_paulis == None or reset == True:
                ops = qt.jmat(self.spin())
                eigs = [op.eigenstates() for op in ops]
                self.precalc_paulis = [ops, eigs]
            return self.precalc_paulis
        else:
            return self.pure_sphere.paulis()

    def eigenenergies(self, reset=False):
        if self.pure_sphere == None:
            if self.precalc_energy_eigs == None and self.energy != None or reset == True:
                self.precalc_energy_eigs = self.energy.eigenstates()
            return self.precalc_energy_eigs
        else:
            return self.pure_sphere.eigenenergies(reset=reset)

    def hermitian_bases(self, reset=False):
        if self.pure_sphere == None:
            if self.precalc_bases == None or reset == True:
                self.precalc_bases = [[qt.Qobj(basis), qt.Qobj(basis).eigenstates()]\
                    for basis in gellman.get_basis(self.n())]
            return self.precalc_bases
        else:
            return self.pure_sphere.hermitian_bases(reset=reset)

    def hermitian_basis(self):
        if self.pure_sphere == None:
            bases = self.hermitian_bases()
            vector = [qt.expect(basis[0], self.state) for basis in bases]
            return vector, bases
        else:
            return self.pure_sphere.hermitian_basis()
   
    def coherent_states(self, N=25, reset=False):
        if self.pure_sphere == None:
            if self.precalc_coherents == None or reset == True:
                theta = np.linspace(0, math.pi, N)
                phi = np.linspace(0, 2*math.pi, N)
                THETA, PHI = np.meshgrid(theta, phi)
                self.precalc_coherents = [[qt.spin_coherent(self.spin(), THETA[i][j], PHI[i][j])\
                                for j in range(N)] for i in range(N)], THETA, PHI
            return self.precalc_coherents 
        else:
            return self.pure_sphere.coherent_states(N=N, reset=reset)

    def husimi(self):
        if self.pure_sphere == None:
            N = 25
            coherents, THETA, PHI = self.coherent_states(N=N)
            Q = np.zeros_like(THETA)
            for i in range(N):
                for j in range(N):
                    probability = (coherents[i][j].ptrace(0)*self.state).tr().real
                    Q[-1*i][-1*j] = probability
            pts = []
            for i, j, k in zip(Q, THETA, PHI):
                for q, t, p in zip(i, j, k):
                    pts.append([q, sph_xyz(t, p)])
            return pts
        else:
            return self.pure_sphere.husimi()

    def husimi_old(self):
        if self.pure_sphere == None:
            N = 25
            theta = np.linspace(0, math.pi, N)
            phi = np.linspace(0, 2*math.pi, N)
            Q, THETA, PHI = qt.spin_q_function(self.state, theta, phi)
            pts = []
            for i, j, k in zip(Q, THETA, PHI):
                for q, t, p in zip(i, j, k):
                    pts.append([q, sph_xyz(t, p)])
            return pts
        else:
            return self.pure_sphere.husimi_old()

    def distinguishable_pieces(self):
        if self.pure_sphere == None:
            if self.dimensionality != None:
                the_state = self.state.copy()
                #print(the_state)
                #print(self.dimensionality)
                the_state.dims = [self.dimensionality, self.dimensionality]
                pieces = [the_state.ptrace(i) for i in range(len(self.dimensionality))]
                return pieces
            else:
                return [self.state]
        else:
            return self.pure_sphere.distinguishable_pieces()

    def dist_pieces_spin(self, pieces):
        if self.pure_sphere == None:
            arrows = []
            for piece in pieces:
                j = dim_spin(piece.shape[0]) 
                direction = np.array([qt.expect(qt.jmat(j, "x"), piece).real,\
                                      -1*qt.expect(qt.jmat(j, "y"), piece).real,\
                                      -1*qt.expect(qt.jmat(j, "z"), piece).real])
                spin_squared = np.sqrt(np.sum(direction**2))
                arrows.append([normalize(direction).tolist(), spin_squared])
            return arrows
        else:
            return self.pure_sphere.dist_pieces_spin(pieces)

    def are_separable(self, pieces):
        if self.pure_sphere == None:
            seps = []
            for piece in pieces:
                entropy = qt.entropy_vn(piece) 
                if entropy < 0.001 and entropy > -0.001:
                    seps.append(True)
                else:
                    seps.append(False)
            return seps
        else:
            return self.pure_sphere.are_separable(pieces)

    def separable_skies(self, pieces, are_separable):
        if self.pure_sphere == None:
            skies = {}
            for i in range(len(pieces)):
                if are_separable[i] == True:
                    q = density_to_purevec(pieces[i])
                    skies[i] = q_SurfaceXYZ(q)
            return skies
        else:
            return self.pure_sphere.separable_skies(pieces, are_separable)

##################################################################################################################

    def set_dimensionality(self, dims):
        self.dimensionality = dims
        if self.pure_sphere != None:
            self.pure_sphere.dimensionality = dims

    def refresh(self, pure=False):
        new_state = self.parent.child_state(self)
        #print("&*&*")
        #print(self.parent.state)
        #print(self.parent.dims)
        #print(new_state)
        #print("23847")
        if self.state != None and new_state.shape[0] != self.state.shape[0]:   
            self.precalc_bases = None
            self.precalc_paulis = None
            self.precalc_energy_eigs = None
            self.precalc_coherents = None
            self.energy = qt.rand_herm(new_state.shape[0])
            if self.pure_sphere != None:
                self.pure_sphere.precalc_bases = None
                self.pure_sphere.precalc_paulis = None
                self.pure_sphere.precalc_energy_eigs = None
                self.pure_sphere.precalc_coherents = None
                self.pure_sphere.dimensionality = None
            self.dimensionality = None
        self.state = new_state
        if self.pure_sphere != None:
            self.pure_sphere.dt = self.dt
            self.pure_sphere.evolving = self.evolving
            self.pure_sphere.dimensionality = self.dimensionality
            self.pure_sphere.energy = self.energy
            self.pure_sphere.evolution = self.evolution
        if pure == False:
            if self.parent.is_separable(self):
                purevec = density_to_purevec(self.state)
                #print("23432")
                #print(purevec)
                #print("2345236")
                if self.pure_sphere == None:
                    self.pure_sphere = PureSphere(state=purevec,\
                                                 energy=self.energy,\
                                                 dt=self.dt,\
                                                 evolving=self.evolving,
                                                 evolution=self.evolution,
                                                 double=self)
                    self.pure_sphere.dimensionality = self.dimensionality
                else:
                    self.pure_sphere.state = purevec
                    self.pure_sphere.dt = self.dt
                    self.pure_sphere.evolving = self.evolving
                    self.pure_sphere.dimensionality = self.dimensionality
                    self.pure_sphere.energy = self.energy
                    self.pure_sphere.evolution = self.evolution
                    self.pure_sphere.precalc_bases = None
                    self.pure_sphere.precalc_paulis = None
                    self.pure_sphere.precalc_energy_eigs = None
                    self.pure_sphere.precalc_coherents = None
            else:
                self.pure_sphere = None
        if self.energy.shape[0] != self.state.shape[0]:
            self.energy = qt.rand_herm(self.state.shape[0])
            if self.pure_sphere != None:
                self.pure_sphere.energy = self.energy
        #if self.pure_sphere != None:
            #print("*((&")
            #print(self.pure_sphere.state)
            #print("2983473982")

    def __getattr__(self, attr):
        if self.pure_sphere != None:
            if hasattr(self.pure_sphere, attr):
                return getattr(self.pure_sphere, attr)
        else:
            return None

    def signal_pure_update(self):
        new_state = self.pure_sphere.state
        new_energy = self.pure_sphere.energy
        self.parent.update_child(self, new_state)
        self.energy = new_energy
        self.refresh(pure=True)

    def random_energy(self):
        self.energy = qt.rand_herm(self.n())
        self.eigenenergies(reset=True)
        if self.pure_sphere != None:
            self.pure_sphere.energy = self.energy
    
    def evolve(self, operator, dt=None, inverse=False):
        if dt == None:
            dt = self.dt
        if self.pure_sphere == None:
            self.parent.evolve_child(self, operator, dt=dt, inverse=inverse)
        else:
            #print("((&&(")
            #print(operator)
            #print(self.pure_sphere.state)
            #print("#42")
            self.pure_sphere.evolve(operator, dt=dt, inverse=inverse)

##################################################################################################################

    def rotate(self, pole, inverse=False):
        if self.pure_sphere == None:
            if pole == "x":
                self.evolve(self.paulis()[0][0], dt=self.dt, inverse=inverse)
            elif pole == "y":
                self.evolve(self.paulis()[0][1], dt=self.dt, inverse=inverse)
            elif pole == "z":
                self.evolve(self.paulis()[0][2], dt=self.dt, inverse=inverse)
            self.refresh()
        else:
            self.pure_sphere.rotate(pole, dt=self.dt, inverse=inverse)

    def rotate_distinguishable(self, i, direction, dt=None, inverse=False):
        if dt == None:
            dt = self.dt
        if self.pure_sphere == None:
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
                self.parent.evolve_child(self, total_op, dt=dt, inverse=inverse)
                self.refresh()
        else:
            self.pure_sphere.rotate_distinguishable(i, direction, dt=dt, inverse=inverse)

##################################################################################################################

    def collapse(self, operator):
        if self.pure_sphere == None:
            stuff = self.parent.collapse_child(self, operator)
            self.refresh()
            return stuff
        else:
            return self.pure_sphere.collapse(operator)

    def distinguishable_collapse(self, i, direction):
        if self.pure_sphere == None:
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
                stuff = self.parent.collapse_child(self, total_op)
                self.refresh()
                return stuff
        else:
            return self.pure_sphere.distinguishable_collapse(i, direction)

##################################################################################################################

    def pretty_state(self):
        if self.pure_sphere == None:
            return np.array_str(self.state.full(),  max_line_width=500, precision=2, suppress_small=True)
        else:
            return self.pure_sphere.pretty_state()

    def pretty_measurements(self, harmonic1D=False, harmonic2D=False):
        if self.pure_sphere != None:
            return self.pure_sphere.pretty_measurements(harmonic1D=harmonic1D, harmonic2D=harmonic2D)
        else:
            s = ""
            ops, eigs = self.paulis()
            signs = ["X", "Y", "Z"]
            keys = ["f", "g", "h"]
            for i in range(len(ops)):
                op = ops[i]
                L, V = eigs[i]
                s += "     %s '%s': %.2f\n" % (signs[i], keys[i], qt.expect(op, self.state))
                for j in range(len(V)):
                    probability = (V[j].ptrace(0)*self.state).tr().real
                    s += "\t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "     H 'y': %.2f (energy)\n" % (qt.expect(self.energy, self.state))
            L, V = self.eigenenergies()
            for j in range(len(V)):
                probability = (V[j].ptrace(0)*self.state).tr().real
                s += "\t%.2f\t%.2f%%\n" % (L[j], probability*100)
            s += "     R 't' (random)\n"
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
                prob = (v.ptrace(0)*self.state).tr().real
                s += "\t%.2f : %.2f\n" % (L[j].real, prob)
        return s[:-1]