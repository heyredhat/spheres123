import sys
import math
import cmath
import gellman
import qutip as qt
import numpy as np
from magic import *
from mixedsphere import *

class Spheres:
    def __init__(self):
        self.state = None
        self.children = []
        self.dims = []

    def pretty_penrose(self):
        if len(self.children) > 1:
            ##print('hiiii')
            s = "   -------------------------------\n"
            for i, a in enumerate(self.children):
                s += "     %d:\n" % i
                for j, b in enumerate(self.children):
                    if i != j:
                        pa = self.penrose_angle(a, b)
                        if pa == None:
                            s += "        has no angle w/ %d\n" % (j)
                        else:
                            s += "        makes %.2f degrees w/ %d\n" % (math.degrees(pa), j)
            return s

    def pretty_children(self, selected):
        s = ""
        for i, child in enumerate(self.children):
            j = dim_spin(self.dims[i])
            s += "     (%d) %.1f\n" % (i, j)
        s += "     \t(this is %d)" % (self.children.index(selected))
        return s

    def add_child(self, child_state):
        ##print("adding child")
        ##print(self.state)
        ##print(self.dims)
        child_state.dims = [[child_state.shape[0]],[1]]
        if self.state == None:
            self.state = child_state
        else:
            self.state = qt.tensor(self.state, child_state)
        self.dims.append(child_state.shape[0])
        self.children.append(MixedSphere(parent=self, energy=qt.rand_herm(child_state.shape[0]),\
                                            dt=0.001, evolving=False))
        #self.children[-1].refresh()
        for child in self.children:
            child.refresh()
        ####print("**")
        ###print(self.state)
        ###print(self.dims)
        return self.children[-1]

    def update_child(self, child, new_state):
        ###print("ha") #PROBLEMS HERE
        if len(self.children) == 1:
            self.state = new_state
            self.dims = new_state.dims[0]
            self.state.dims = [self.dims, [1]*len(self.dims)]
        else:
            if self.is_separable(child):
                temp_state = self.state.copy()
                i = self.children.index(child)
                temp_dims = self.dims[:]
                temp_state = qt.tensor_swap(temp_state, (i, 0))
                temp_dims[i], temp_dims[0] = temp_dims[0], temp_dims[i]
                temp_state = density_to_purevec(temp_state.ptrace(list(range(1,len(self.dims)))))
                temp_state = qt.tensor(new_state, temp_state)
                temp_dims[0] = new_state.shape[0]
                temp_state.dims = [temp_dims, [1]*len(temp_dims)]
                temp_state = qt.tensor_swap(temp_state, (0, i))
                self.state = temp_state
                self.dims[i] = new_state.shape[0]
                self.state.dims = [self.dims, [1]*len(self.dims)]
                ###print('hi')
                ##print("(((")
                ##print(self.state)
                ##print(self.dims)
                ##print("_))")
                child.refresh(pure=True)

    def remove_child(self, child_sphere):
        if len(self.children) == 0:
            return
        elif len(self.children) == 1:
            child_sphere.state = self.state
            self.children = []
            self.dims = []
            self.state = None
            child_sphere.refresh()
        else:
            if self.is_separable(child_sphere):
                pure_sphere = PureSphere(state=density_to_purevec(child_sphere.state),\
                                         energy=child_sphere.energy,\
                                         dt=child_sphere.dt,\
                                         evolving=child_sphere.evolving)
                pure_sphere.dimensionality = child_sphere.dimensionality
                temp_state = density_to_purevec(self.other_state(child_sphere))
                i = self.children.index(child_sphere)
                del self.children[i]
                del self.dims[i]
                self.state = temp_state
                return pure_sphere

    def is_separable(self, child_sphere):
        return mixed_separable(self.child_state(child_sphere))

    def child_state(self, child_sphere):
        if len(self.children) == 1:
            return self.state.ptrace(0)
        else:
            #print(child_sphere)
            #print(self.children)
            #print(self.state)
            #sys.stdout.flush()
            if child_sphere in self.children:
                i = self.children.index(child_sphere)
                return self.state.ptrace(i)

    def other_state(self, child_sphere):
        if child_sphere in self.children:
            i = self.children.index(child_sphere)
            keep = list(range(len(self.children)))
            keep.remove(i)
            return self.state.ptrace(keep)

    def children_state(self, children):
        indices = []
        for child in children:
            if child not in self.children:
                return
            else:
                indices.append(self.children.index(child))
        return self.state.ptrace(indices)

    def other_children_state(self, children):
        if len(children) == len(self.children):
            return
        else:
            indices = []
            for child in children:
                if child not in self.children:
                    return
                else:
                    indices.append(self.children.index(child))
            everyone_else = list(range(len(self.children)))
            everyone_else = [i for j, i in enumerate(everyone_else) if j not in indices]
            return self.state.ptrace(everyone_else)

    def upgrade_operator(self, child_sphere, operator):
        if child_sphere in self.children:
            i = self.children.index(child_sphere)
            op = operator if i == 0 else qt.identity(self.dims[0])
            for j in range(1, len(self.children)):
                op = qt.tensor(op, operator) if j == i else qt.tensor(op, qt.identity(self.dims[j]))
            return op

    def evolve_child(self, child_sphere, operator, dt=0.01, inverse=False):
        if child_sphere in self.children:
            #print("operator")
            #print(operator)
            op = self.upgrade_operator(child_sphere, operator) if len(self.children) > 1 else operator
            #print("op")
            #print(op)
            unitary = (-2*math.pi*1j*op*dt).expm()
            if inverse:
                unitary = unitary.dag()
            cp = self.state.copy()
            cp.dims = [self.dims, [1]*len(self.dims)]
            unitary.dims = [self.dims, self.dims]
            #print("(")
            #print(unitary)
            #print(cp)
            #print(")")
            cp = unitary*cp
            self.state = cp

            child_sphere.refresh()

    def collapse_child(self, child_sphere, operator):
        if child_sphere in self.children:
            op = self.upgrade_operator(child_sphere, operator) if len(self.children) > 1 else operator
            L, V = op.eigenstates()
            amplitudes = [self.state.overlap(v) for v in V]
            probabilities = np.array([(a*np.conjugate(a)).real for a in amplitudes])
            probabilities = probabilities/probabilities.sum()
            pick = np.random.choice(list(range(len(V))), 1, p=probabilities)[0]
            vec = V[pick].full().T[0]
            projector = qt.Qobj(np.outer(vec,np.conjugate(vec)))
            projector.dims = [self.dims, self.dims]
            self.state = (projector*self.state).unit()
            child_sphere.refresh()
            return L[pick], L, probabilities

    def collide_children(self, a, b):
        if a in self.children and b in self.children:
            AA = self.child_state(a)
            spinA = (AA.shape[0]-1)/2.
            BB = self.child_state(b)
            spinB = (BB.shape[0]-1)/2.

            #print("state:\n%s" % self.state)

            ai = self.children.index(a)
            bi = self.children.index(b)

            #print("child index ai %d" % ai)
            #print("child index bi %d" % bi)

            temp_state = self.state.copy()

            temp_state = qt.tensor_swap(temp_state, (ai, 0))
            if bi == 0:
                bi = ai
            temp_state = qt.tensor_swap(temp_state, (bi, 1))

            #print("swapped state\n%s" % temp_state)

            temp_dims = self.dims[:]
            temp_dims[0], temp_dims[ai] = temp_dims[ai], temp_dims[0]
            temp_dims[1], temp_dims[bi] = temp_dims[bi], temp_dims[1]

            #print("swapped dims %s" % temp_dims)

            #print("spinA %f" % spinA)
            #print("spinB %f" % spinB)

            OPERATOR, STATES, GOTO, WHICH_IS = coupling(spinA, spinB)

            #print("coupling")
            #print("operator")
            #print(OPERATOR)
            #print("which_is %s" % WHICH_IS)

            boundaries = [0]
            boundaries_are = [WHICH_IS[0][0]]
            last_which = WHICH_IS[0][0]
            for i in range(len(WHICH_IS)):
                if WHICH_IS[i][0] != last_which:
                    boundaries.append(i)
                    boundaries_are.append(WHICH_IS[i][0])
                last_which = WHICH_IS[i][0]

            #print("boundaries %s" % boundaries)
            #print("boundaries are %s" % boundaries_are)

            #print("{")
            #print(temp_dims)
            for state in STATES:
                state.dims = [[state.shape[0]], [1]]
            #print(spinA)
            #print(spinB)
            #print(STATES)
            projectors = [qt.Qobj(state).ptrace(0) for state in STATES]
            #print(projectors)
            upgraded_projectors = []
            for i in range(len(STATES)):
                total_op = projectors[i]
                for j in range(2, len(temp_dims)):
                    total_op = qt.tensor(total_op, qt.identity(temp_dims[j]))
                total_op.dims = [[total_op.shape[0]], [total_op.shape[0]]]
                upgraded_projectors.append(total_op)
            #print(upgraded_projectors)
            #print("}")
            #print("projectors")
            #print(upgraded_projectors)

            temp_state.dims = [[temp_state.shape[0]], [1]]
            temp_dm = temp_state.ptrace(0)
            temp_state.dims = [temp_dims, [1]*len(temp_dims)]
            temp_dm.dims = [[temp_dm.shape[0]], [temp_dm.shape[0]]]
            #print("(")
            #print(upgraded_projectors[0].dims)
            #print(temp_dm.dims)
            #print(")")
            probabilities = [(proj*temp_dm).tr().real for proj in upgraded_projectors]

            #print("which_is %s" % WHICH_IS)
            #print("probabilities %s" % probabilities )
            #print(probabilities)

            for i in range(len(WHICH_IS)):
                j, m = WHICH_IS[i]
                if j == 0: # No spin-0 for now
                    probabilities[i] = 0

            probabilities = np.array(probabilities)
            probabilities = probabilities/probabilities.sum()

            j_probs = []
            for i, b in enumerate(boundaries):
                if i == len(boundaries)-1:
                    j_probs.append(sum(probabilities[b:]))
                else:
                    j_probs.append(sum(probabilities[b:boundaries[i+1]]))

            #print("reformed probabilities over j's")
            #print(j_probs)

            pick = np.random.choice(list(range(len(j_probs))), 1, p=j_probs)[0]

            #print("pick %d" % pick)

            cutA = int(boundaries[pick])
            cutB = int(cutA + boundaries_are[pick]*2 + 1)

            #print("cutting from %d to %d of %s" % (cutA, cutB, WHICH_IS))

            REFHALF = STATES[cutA:cutB]
            ONE = STATES[:cutA]
            TWO = STATES[cutB:]
            STATES = np.array(ONE+TWO+REFHALF)

            #print("len(ONE)=%d + len(TWO)=%d + len(REFHALF)=%d = %d" % (len(ONE), len(TWO), len(REFHALF), len(STATES)))

            beginna = int(len(ONE+TWO)*np.prod(temp_dims[2:]))
            #print("cutting from %d" % beginna)

            GREFHALF = GOTO[cutA:cutB]
            GONE = GOTO[:cutA]
            GTWO = GOTO[cutB:]
            GOTO = np.array(GONE+GTWO+GREFHALF)

            WREFHALF = WHICH_IS[cutA:cutB]
            WONE = WHICH_IS[:cutA]
            WTWO = WHICH_IS[cutB:]
            WHICH_IS = WONE+WTWO+WREFHALF

            OPERATOR = qt.Qobj(np.array([state.full().T[0] for state in STATES]))

            #print("reorganized operator")
            #print(OPERATOR)

            total_op = OPERATOR
            for i in range(2, len(self.children)):
                total_op = qt.tensor(total_op, qt.identity(temp_dims[i]))

            total_op.dims = [[total_op.shape[0]], [total_op.shape[1]]]
            temp_state.dims = [[temp_state.shape[0]], [1]]

            #print("total_op")
            #print(total_op)

            temp_state = total_op*temp_state

            #print("projected state")
            #print(temp_state)

            temp_state = qt.Qobj( np.array( temp_state.full().T[0].tolist()[beginna:] ) ).unit()
            
            #print("truncated state")
            #print(temp_state)
            #print(temp_dims)

            del temp_dims[0]
            del temp_dims[0]
            final_j = boundaries_are[pick]
            temp_dims.insert(0, int(2*final_j+1))

            #print("new_dims")
            #print(temp_dims)

            temp_state.dims = [temp_dims, [1]*len(temp_dims)]
            stuff = [boundaries_are[pick], np.array(boundaries_are), np.array(j_probs)]
            return (ai, bi, temp_state, temp_dims, stuff)

    def finish_up_collide(self, ai, bi, temp_state, temp_dims):
        ##print("finishing up collide")
        self.children[0] = self.children[ai]
        self.children[1] = self.children[bi]

        del self.children[0]
        del self.children[0]

        self.children.insert(0, MixedSphere(parent=self, energy=qt.rand_herm(temp_dims[0])))

        self.state = temp_state
        self.dims = temp_dims
        self.state.dims = [self.dims, [1]*len(self.dims)]
        ##print("@@@@@")
        ##print(self.children[0].state)
        ##print(self.children[0].energy)
        ##print("@")
        self.children[0].refresh()
        ##print(self.children[0].state)
        ##print(self.children[0].energy)
        ##print("****")
        return self.children[0]

    def split_child(self, child, spin_a, spin_b):
        if child in self.children:
            my = self.child_state(child)
            my_spin = (my.shape[0]-1)/2.

            ##print("state:\n%s" % self.state)

            ci = self.children.index(child)

            ##print("child index %d" % ci)

            temp_state = self.state.copy()
            temp_state = qt.tensor_swap(temp_state, (ci, 0))

            temp_dims = self.dims[:]
            temp_dims[0], temp_dims[ci] = temp_dims[ci], temp_dims[0]

            ##print("swapped dims %s" % temp_dims)

            OPERATOR, STATES, GOTO, WHICH_IS = coupling(spin_a, spin_b)

            ##print("orig which_is")
            ##print(WHICH_IS)

            FSTATES = []
            FGOTO = []
            FWHICH_IS = []
            for i in range(len(WHICH_IS)):
                if WHICH_IS[i][0] == my_spin:
                    FSTATES.append(STATES[i])
                    FGOTO.append(GOTO[i])
                    FWHICH_IS.append(WHICH_IS[i])

            ##print("-> states")
            ##print(FSTATES)
            ##print("-> goto")
            ##print(FGOTO)
            ##print("-> which_is")
            ##print(FWHICH_IS)

            FOPERATOR = qt.Qobj(np.array([state.full().T[0] for state in FSTATES]))
            ##print("-> operator")
            ##print(FOPERATOR)

            total_op = FOPERATOR
            for i in range(1, len(self.children)):
                total_op = qt.tensor(total_op, qt.identity(temp_dims[i]))
            total_op = total_op.dag()
            ##print("-> upgraded operator")
            ##print(total_op)

            ##print(temp_state)

            ##print("-> on state")
            temp_state = total_op*temp_state
            ##print(temp_state)

            temp_dims.insert(0, int(spin_a*2 + 1))
            temp_dims[1] = int(spin_b*2 + 1)
            temp_state.dims = [temp_dims, [1]*len(temp_dims)]

            ##print("new dims %s" % temp_dims)

            return (temp_dims, temp_state)

    def finish_up_split(self, temp_dims, temp_state):
        ##print("finishing up split")
        del self.children[0]
        self.dims = temp_dims
        self.state = temp_state
        self.state.dims = [self.dims, [1]*len(self.dims)]
        ##print("(*&*&#^$")
        ##print(self.state)
        self.children.insert(0, MixedSphere(parent=self,energy=qt.rand_herm(self.dims[1])))
        self.children.insert(0, MixedSphere(parent=self,energy=qt.rand_herm(self.dims[0])))
        self.children[0].refresh()
        self.children[1].refresh()
        return self.children[0]

    def penrose_angle(self, a, b, twice=None):
        if a in self.children and b in self.children:
            AA = self.child_state(a)
            spinA = (AA.shape[0]-1)/2.
            if spinA <= 0.5:
                return None
            else:
                #print("uaefhiuawehfuaewhf")
                BB = self.child_state(b)
                spinB = (BB.shape[0]-1)/2.

                #print(BB)
                #print(spinB)

                ai = self.children.index(a)

                #print(ai)

                #print(self.state)

                temp_state = self.state.copy()
                temp_state = qt.tensor_swap(temp_state, (ai, 0))
                temp_dims = self.dims[:]
                temp_dims[0], temp_dims[ai] = temp_dims[ai], temp_dims[0]

                #print(temp_state)
                #print(temp_dims)

                OPERATOR, STATES, GOTO, WHICH_IS = coupling(spinA-0.5, 0.5)

                FSTATES = []
                FGOTO = []
                FWHICH_IS = []
                for i in range(len(WHICH_IS)):
                    if WHICH_IS[i][0] == spinA:
                        FSTATES.append(STATES[i])
                        FGOTO.append(GOTO[i])
                        FWHICH_IS.append(WHICH_IS[i])
                FOPERATOR = qt.Qobj(np.array([state.full().T[0] for state in FSTATES]))

                total_op = FOPERATOR
                for i in range(1, len(self.children)):
                    total_op = qt.tensor(total_op, qt.identity(temp_dims[i]))
                total_op = total_op.dag()

                temp_state = total_op*temp_state
                temp_dims.insert(0, int((spinA-0.5)*2 + 1))
                temp_dims[1] = int((0.5)*2 + 1)
                temp_state.dims = [temp_dims, [1]*len(temp_dims)]
                #print("(((")
                #print(temp_dims)
                #print(temp_state)
                #print(")))")

                bi = self.children.index(b)
                if bi == 0:
                    bi = ai
                bi += 1
                ei = 1

                #print("bi")
                #print(bi)

                temp_state = qt.tensor_swap(temp_state, (bi, 0))
                temp_dims[0], temp_dims[bi] = temp_dims[bi], temp_dims[0]

                OPERATOR, STATES, GOTO, WHICH_IS = coupling(spinB, 0.5)

                boundaries = [0]
                boundaries_are = [WHICH_IS[0][0]]
                last_which = WHICH_IS[0][0]
                for i in range(len(WHICH_IS)):
                    if WHICH_IS[i][0] != last_which:
                        boundaries.append(i)
                        boundaries_are.append(WHICH_IS[i][0])
                    last_which = WHICH_IS[i][0]

                for state in STATES:
                    state.dims = [[state.shape[0]], [1]]
                projectors = [qt.Qobj(state).ptrace(0) for state in STATES]
                #print("{")
                ##print(temp_state)
                ##print(projectors[0])
                #print(temp_dims)
                #print(projectors[0])
                #print("7")
                upgraded_projectors = []
                for i in range(len(STATES)):
                    total_op = projectors[i]
                    for j in range(2, len(temp_dims)):
                        total_op = qt.tensor(total_op, qt.identity(temp_dims[j]))
                    total_op.dims = [[total_op.shape[0]], [total_op.shape[0]]]
                    upgraded_projectors.append(total_op)

                #print("22")
                #print(temp_state)
                #print("33")
                temp_state.dims = [[temp_state.shape[0]], [1]]
                temp_dm = temp_state.ptrace(0)
                temp_state.dims = [temp_dims, [1]*len(temp_dims)]
                temp_dm.dims = [[temp_dm.shape[0]], [temp_dm.shape[0]]]

                #print("(")
                #print(temp_dm)
                #print(upgraded_projectors[0])
                #print(")")
                probabilities = [(proj*temp_dm).tr().real for proj in upgraded_projectors]

                for i in range(len(WHICH_IS)):
                    j, m = WHICH_IS[i]
                    if j == 0: # No spin-0 for now
                        probabilities[i] = 0

                probabilities = np.array(probabilities)
                probabilities = probabilities/probabilities.sum()

                j_probs = []
                for i, B in enumerate(boundaries):
                    if i == len(boundaries)-1:
                        j_probs.append(sum(probabilities[B:]))
                    else:
                        j_probs.append(sum(probabilities[B:boundaries[i+1]]))

                probs = {"up" : j_probs[0], "down" : j_probs[1]}
                theta = math.acos(2*probs["up"] - 1) #math.acos(1-2*probs["down"])
                return theta


if __name__ == '__main__':
    spheres = Spheres()
    a = spheres.add_child(qt.rand_ket(10))
    b = spheres.add_child(qt.rand_ket(12))
    #spheres.state = qt.rand_ket(6*7)
    #spheres.state.dims = [[6,7], [1,1]]
    #spheres.split_child(a,0.5,0.5)
    spheres.penrose_angle(a,b)
