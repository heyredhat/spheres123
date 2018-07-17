import math
import cmath
import mpmath
import sympy
import scipy
import functools
import qutip as qt
import numpy as np
import gellman
import itertools
import operator

##################################################################################################################

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def sph_xyz(theta, phi):
    return [math.sin(theta)*math.cos(phi),\
            math.sin(theta)*math.sin(phi),\
            math.cos(theta)]

def projectors(operator):
    L, V = operator.eigenstates()
    V = [v.ptrace(0) for v in V]
    return L, V

def evolver(state, operator, dt=0.01, inverse=False):
    unitary = (-2*math.pi*1j*operator*dt).expm()
    if inverse:
        unitary = unitary.dag()
    return unitary*state

def dim_spin(n):
    return (n-1.)/2.

##################################################################################################################

def c_xyz(c):
    if c == float('inf'):
        return [0,0,1]
    x = c.real
    y = c.imag
    return [(2*x)/(1.+(x**2)+(y**2)),\
            (2*y)/(1.+(x**2)+(y**2)),\
            (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]

def xyz_c(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    if z == 1:
        return float('inf') 
    else:
        return complex(x/(1-z), y/(1-z))

def combos(a,b):
    f = math.factorial
    return f(a) / f(b) / f(a-b)

def polynomial_v(coefficients):
    spin = (len(coefficients)-1.)/2.
    terms = []
    i = 0
    for m in np.arange(-1*spin, spin+1, 1):
        term = 1./(math.sqrt(math.factorial(2*spin)/(math.factorial(spin-m)*math.factorial(spin+m))))*coefficients[i]
        terms.append(complex(term))
        i += 1
    return np.array(terms[::-1])

def v_polynomial(v):
    components = v.tolist()
    spin = (len(components)-1.)/2.
    terms = []
    i = 0
    for m in np.arange(-1*spin, spin+1, 1):
        term = math.sqrt(math.factorial(2*spin)/(math.factorial(spin-m)*math.factorial(spin+m)))*components[i]
        terms.append(complex(term))
        i += 1
    return terms[::-1]

def C_polynomial(roots):
    zeros2 = roots.count(complex(0,0))
    if zeros2 == len(roots):
        return [complex(1,0)] + [complex(0,0)]*(zeros2) 
    zeros = roots.count(float('Inf'))
    #zeros = roots.count(complex(0,0))+roots.count(float('Inf'))
    roots = [root for root in roots if root != float('Inf')]
    if len(roots) == 0:
        return [complex(0,0)]*zeros + [complex(1,0)]
    else:
        s = sympy.symbols("s")
        polynomial = sympy.Poly(functools.reduce(lambda a, b: a*b, [s-root for root in roots]), domain="CC")
        if zeros2 > 0:
        #    return [complex(1,0)] + [complex(0,0)]*zeros + [complex(0,0)]*(len(roots))
           #if zeros2 > 1:
            return [complex(0,0)]*(zeros) + [complex(c) for c in polynomial.coeffs()] + [complex(0,0)]*(zeros2) 
            #else:
            #    return [complex(c) for c in polynomial.coeffs()] ++ [complex(0,0)]*(zeros)
        else:
            return [complex(0,0)]*(zeros) + [complex(c) for c in polynomial.coeffs()]

        #print("UU"+ str(zeros2))
        #if 


def polynomial_C(polynomial):
    zeros = 0
    for i in range(len(polynomial)):
        if polynomial[i] == 0:
            zeros +=1
        else:
            break
    poles = [float('Inf') for i in range(zeros)]
    try:
        roots = [complex(root) for root in mpmath.polyroots(polynomial)]
    except:
        try:
            roots = [complex(root) for root in np.roots(polynomial)]
        except:
            print(polynomial)
            roots = [float('Inf') for i in range(len(polynomial)-zeros)]
    return poles+roots




def C_v(roots):
    #print("{")
    #print(roots)
   # print(C_polynomial(roots))
    #print(polynomial_v(C_polynomial(roots)))
    #print("}")
    return polynomial_v(C_polynomial(roots))

def v_C(v):
    return polynomial_C(v_polynomial(v))

def v_SurfaceXYZ(v):
    return [c_xyz(c) for c in v_C(v)]

def SurfaceXYZ_v(XYZ):
    return C_v([xyz_c(xyz) for xyz in XYZ])

def q_SurfaceXYZ(q):
    return v_SurfaceXYZ(q.full().T[0])

def SurfaceXYZ_q(XYZ):
    return qt.Qobj(SurfaceXYZ_v(XYZ)).unit()

##################################################################################################################

def q_qubits(state):
    xyzs = q_SurfaceXYZ(state)
    #print(xyzs)
    return [SurfaceXYZ_q([xyz]) for xyz in xyzs]

def qubits_q(qubits):
    return SurfaceXYZ_q([q_SurfaceXYZ(qubit)[0] for qubit in qubits])

def symmeterize(pieces):
    if (len(pieces)) == 1:
        return pieces[0]
    n = len(pieces)
    normalization = 1./math.factorial(n)
    permutations = list(itertools.permutations(pieces, n))
    perm_states = []
    for permutation in permutations:
        perm_state = permutation[0]
        for state in permutation[1:]:
            perm_state = qt.tensor(perm_state, state)
        perm_state.dims = [[perm_state.shape[0]],[1]]
        perm_states.append(perm_state)
    tensor_sum = sum(perm_states)
    return normalization*tensor_sum

def dicke_states(n):
    states = []
    for k in range(n+1):
        pieces = [qt.basis(2, 0) for i in range(n-k)]
        pieces.extend([qt.basis(2,1) for i in range(k)])
        #print("dicke")
        #print("pieces")
        #print(pieces)
        dicke = math.sqrt(math.factorial(n)/(math.factorial(n-k)*math.factorial(k)))*symmeterize(pieces)
        states.append(dicke)
        #print(dicke)
    return states

def unsymmeterize(state, use_dickes=None):
    n = state.shape[0]
    d = int(math.log(n, 2))
    dickes = dicke_states(d) if use_dickes == None else use_dickes
    amps = [state.overlap(dicke) for dicke in dickes]
    return qt.Qobj(np.conjugate(np.array(amps))).unit()

##################################################################################################################

def mink_hermitianPoint(mink):
    #print(mink)
    t, x, y, z = mink.tolist()
    return (t*qt.identity(2) + x*qt.sigmax() + y*qt.sigmay() + z*qt.sigmaz()).unit()

def hermitianPoint_mink(point):
    return np.array([0.5*qt.expect(qt.identity(2), point),\
            0.5*qt.expect(qt.sigmax(), point),\
            0.5*qt.expect(qt.sigmay(), point),\
            0.5*qt.expect(qt.sigmaz(), point)])

def mink_hermitianPoint_ndim(mink, n):
    t, x, y, z = mink.tolist()
    j = dim_spin(n) 
    return (t*qt.identity(n) + x*qt.jmat(j, "x") + y*qt.jmat(j, "y") + z*qt.jmat(j, "z")).unit()

def hermitianPoint_mink_ndim(point, n):
    j = dim_spin(n) 
    return [qt.expect(qt.identity(n), point),\
            qt.expect(qt.jmat(j, "x"), point),\
            qt.expect(qt.jmat(j, "y"), point),\
            qt.expect(qt.jmat(j, "z"), point)]

def qubit_mink(qubit):
    a, b = qubit.full().T[0].tolist()
    return np.array([(a*np.conjugate(a) + b*np.conjugate(b)).real,\
            2*(a*np.conjugate(b)).real,\
            -2*(a*np.conjugate(b)).imag,\
            (a*np.conjugate(a) - b*np.conjugate(b)).real])

def qubit_hermitianPoint(qubit):
    return qubit.ptrace(0)

def hermitianPoint_qubit(herm):
    #if herm.tr() == (herm*herm).tr():
    xyz = [-1*qt.expect(qt.sigmax(), herm),\
               qt.expect(qt.sigmay(), herm),\
               qt.expect(qt.sigmaz(), herm)]
    return SurfaceXYZ_q([xyz])
    #else:
    #    return "Not pure!"

def mink_qubit(mink):
    herm = mink_hermitianPoint(mink)
    return hermitianPoint_qubit(herm)

# 2x2
def applyMobiusToPoint(mobius, herm):
    return mobius*herm*mobius.dag()

Kx = np.array([[0,1,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0]])
Ky = np.array([[0,0,1,0], [0,0,0,0], [1,0,0,0], [0,0,0,0]])
Kz = np.array([[0,0,0,1], [0,0,0,0], [0,0,0,0], [1,0,0,0]])

Jx = np.array([[0,0,0,0], [0,0,0,0], [0,0,0,-1], [0,0,1,0]])
Jy = np.array([[0,0,0,0], [0,0,0,1], [0,0,0,0], [0,-1,0,0]])
Jz = np.array([[0,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,0]])

def mink_rotate(mink, axis, angle, inverse=False):
    angle = angle*10
    op = None
    if axis == "x":
        op = scipy.linalg.expm(angle*Jx)
    elif axis == "y":
        op = scipy.linalg.expm(angle*Jy)
    elif axis == "z":
        op = scipy.linalg.expm(angle*Jz)
    if inverse:
        op = np.linalg.inv(op)
    return np.inner(op,mink)

def mink_boost(mink, axis, rapidity, inverse=False):
    rapidity = rapidity*10
    op = None
    if axis == "x":
        op = scipy.linalg.expm(-1*rapidity*Kx)
    elif axis == "y":
        op = scipy.linalg.expm(-1*rapidity*Ky)
    elif axis == "z":
        op = scipy.linalg.expm(-1*rapidity*Kz)
    if inverse:
        op = np.linalg.inv(op)
    return np.inner(op,mink)

def mink_boost_state(state, axis, dt=0.01, inverse=False):
    qubits = q_qubits(state)
    minks = [mink_boost(qubit_mink(qubit), axis, dt, inverse=inverse) for qubit in qubits]
    qubits2 = [mink_qubit(mink) for mink in minks]
    return qubits_q(qubits2)

def mink_rotate_state(state, axis, dt=0.01, inverse=False):
    qubits = q_qubits(state)
    minks = [mink_rotate(qubit_mink(qubit), axis, dt, inverse=inverse) for qubit in qubits]
    qubits2 = [mink_qubit(mink) for mink in minks]
    return qubits_q(qubits2)

##################################################################################################################

def separable(whole, dims, piece_index):
    whole_copy = whole.copy()
    whole_copy.dims = [[dims],[1]*len(dims)]
    reduction = whole_copy.ptrace(piece_index)
    entropy = qt.entropy_vn(reduction) 
    if entropy < 0.000001 and entropy > -0.999999:
        return True
    else:
        return False

def fuzzy(a, b):
    if a-b < 0.001 and a-b > -0.001:
        return True
    else:
        return False

def density_to_purevec(density):
    entropy = qt.entropy_vn(density) 
    if fuzzy(entropy, 0):
        U, S, V = np.linalg.svd(density.full())
        #print("{")
        #print(U)
        #print(S)
        #print(V)
        #print("}")
        s = S.tolist()
        for i in range(len(s)):
            if fuzzy(s[i], 1):
                return qt.Qobj(np.conjugate(V[i]))

#if __name__ == "__main__":
#    pass
    #a = qt.rand_ket(2)
    #b = a.ptrace(0)
    #c = density_to_purevec(b)

##################################################################################################################

def apply_mobius(state, mobius):
    points = [qubit_hermitianPoint(qubit) for qubit in q_qubits(state)]
    print("{")
    print(points)
    mob = qt.Qobj(mobius)
    points = [mob.dag()*point*mob for point in points]
    print(points)
    return qubits_q([hermitianPoint_qubit(point) for point in points])

def mobius_connection(three_stars_now, three_stars_later):
    abc = [xyz_c(xyz) for xyz in three_stars_now]
    xyz = [xyz_c(xyz) for xyz in three_stars_later]

    A = np.array([["ax", "x", 1],
                  ["by", "y", 1],
                  ["cz", "z", 1]])
    A_ = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]], dtype=np.complex128)
    B = np.array([["ax", "a", "x"],
                  ["by", "b", "y"],
                  ["cz", "x", "z"]])
    B_ = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]], dtype=np.complex128)
    C = np.array([["a", "x", 1],
                  ["b", "y", 1],
                  ["c", "z", 1]])
    C_ = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]], dtype=np.complex128)
    D = np.array([["ax", "a", 1],
                  ["by", "b", 1],
                  ["cz", "c", 1]])
    D_ = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]], dtype=np.complex128)

    ABC = ["a", "b", "c"]
    XYZ = ["x", "y", "z"]
    GUYS = [A, B, C, D]
    GUYS_ = [A_, B_, C_, D_]

    for m in range(3):
        if abc[m] == float('inf'):
            for i in range(3):
                for j in range(3):
                    for k in range(4):
                        if isinstance(GUYS[k][i][j], str):
                            if GUYS[k][i][j].contains(ABC[m]):
                                GUYS[k][i][j].remove(ABC[m])
                            else:
                                GUYS[k][i][j] = '0'
        if xyz[m] == float('inf'):
            for i in range(3):
                for j in range(3):
                    for k in range(4):
                        if isinstance(GUYS[k][i][j], str):
                            if GUYS[k][i][j].contains(XYZ[m]):
                                GUYS[k][i][j].remove(XYZ[m])
                            else:
                                GUYS[k][i][j] = '0'

    for i in range(3):
        for j in range(3):
            for k in range(4):
                if isinstance(GUYS[k][i][j], str):
                    if len(GUYS[k][i][j]) == 1:
                        if GUYS[k][i][j] == '0':
                            GUYS_[k][i][j] = 0
                        elif GUYS[k][i][j] in ABC:
                            GUYS_[k][i][j] = abc[ABC.index(GUYS[k][i][j])]
                        elif GUYS[k][i][j] in XYZ:
                            GUYS_[k][i][j] = xyz[XYZ.index(GUYS[k][i][j])]
                    else:
                       # print(GUYS[k][i][j])
                        one, two = GUYS[k][i][j][0], GUYS[k][i][j][1]
                        if one in ABC:
                            one = abc[ABC.index(one)]
                        elif one in XYZ:
                            one = xyz[XYZ.index(one)]
                        if two in ABC:
                            two = abc[ABC.index(two)]
                        elif two in XYZ:
                            two = xyz[XYZ.index(two)]
                        GUYS_[k][i][j] = one*two

    #print(GUYS_[0])

    AA = np.linalg.det(GUYS_[0])
    BB = np.linalg.det(GUYS_[1])
    CC = np.linalg.det(GUYS_[2])
    DD = np.linalg.det(GUYS_[3])
    return np.array([[AA, BB], [CC, DD]])    

if __name__ == "__main__":
    stateA = qt.rand_ket(4)
    stateB = qt.rand_ket(4)
    stateAstars = q_SurfaceXYZ(stateA)
    stateBstars = q_SurfaceXYZ(stateB)
    mob = mobius_connection(stateAstars, stateBstars)
    Atransformed = apply_mobius(stateA, mob)
    AtransformedStars = q_SurfaceXYZ(Atransformed)

def mobius_connected(skyA, skyB):
    pass

def mobius_for_collapse():
    pass

def decompose_mobius_rot_boost():
    pass



def iterate_majorana():
    pass

##################################################################################################################

def factors(n):    
    return set(functools.reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

##################################################################################################################

if __name__ == '__main2__':
    herm = qt.rand_herm(2)
    mink = hermitianPoint_mink(herm)
    herm2 = mink_hermitianPoint(mink)

    aqubit = qt.rand_ket(2)
    amink = qubit_mink(aqubit)
    aherm = mink_hermitianPoint(amink)
    aherm2 = qubit_hermitianPoint(aqubit)
    aqubit2 = hermitianPoint_qubit(aherm)
    amink2 = qubit_mink(aqubit2)

if __name__ == '__another__':
    n = 4
    state = qt.rand_ket(n)
    print("state")
    print(state)
    qubits = q_qubits(state)
    print("qubits")
    print(qubits)
    state2 = qubits_q(qubits)
    #print("back to state")
    #print(state2)
    #print(q_qubits(state2))
    sym = symmeterize(qubits)
    print("symmeterized")
    print(sym)
    state2 = unsymmeterize(sym)
    print("back to state")
    print(state2)
    qubits2 = q_qubits(state2)
    print("back to qubits")
    print(qubits2)

    #qubit = qt.rand_ket(2)
    #xyz = q_SurfaceXYZ(qubit)
    #new_qubit = SurfaceXYZ_q(xyz)
    #xyz2 = q_SurfaceXYZ(new_qubit)

    #v = qubit.full().T[0]
    #polynomial = v_polynomial(v)
    #v2 = polynomial_v(polynomial)

    #C = polynomial_C2(polynomial) 
    #poly2 = C_polynomial2(C)


