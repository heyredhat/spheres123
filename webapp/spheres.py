import math
import mpmath
import sympy
import functools
import qutip as qt
import numpy as np

##################################################################################################################

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def c_xyz(c):
    if c == float('inf'):
        return [0,0,1]
    x = c.real
    y = c.imag
    return [-1*(2*x)/(1.+(x**2)+(y**2)),\
            (2*y)/(1.+(x**2)+(y**2)),\
            (-1.+(x**2)+(y**2))/(1.+(x**2)+(y**2))]

def xyz_c(xyz):
    x, y, z = -1*xyz[0], xyz[1], xyz[2]
    if z == 1:
        return float('inf') 
    else:
        return complex(x/(1-z), y/(1-z))

def combos(a,b):
    f = math.factorial
    return f(a) / f(b) / f(a-b)

def polynomial_v(polynomial):
    coordinates = [polynomial[i]/(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) for i in range(len(polynomial))]
    return np.array(coordinates)

def v_polynomial(v):
    polynomial = v.tolist()
    return [(((-1)**i) * math.sqrt(combos(len(polynomial)-1,i))) * polynomial[i] for i in range(len(polynomial))] 

def C_polynomial(roots):
    s = sympy.symbols("s")
    polynomial = sympy.Poly(functools.reduce(lambda a, b: a*b, [s-np.conjugate(root) for root in roots]), domain="CC")
    return [complex(c) for c in polynomial.coeffs()]

def polynomial_C(polynomial):
    try:
        roots = [np.conjugate(complex(root)) for root in mpmath.polyroots(polynomial)]
    except:
        return [float('Inf') for i in range(len(polynomial)-1)]
    return roots

def C_v(roots):
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
    return qt.Qobj(C_v([xyz_c(xyz) for xyz in XYZ]))

##################################################################################################################

class Sphere:
    def __init__(self, state=None,\
                       energy=None,\
                       dt=0.01,\
                       evolving=False):
        self.state = state if state != None else qt.rand_ket(2)
        self.energy = energy if energy != None else qt.rand_herm(2)

        self.dt = dt
        self.evolving = evolving

    def n(self):
        return self.state.shape[0]

    def random_state(self):
        self.state = qt.rand_ket(self.n())

    def random_energy(self):
        self.energy = qt.rand_herm(self.n())

    def spin(self):
        return (self.n()-1.)/2.

    def spin_axis(self):
        mink = np.array([qt.expect(qt.identity(self.n()), self.state),\
                         qt.expect(qt.jmat(self.spin(), "x"), self.state),\
                         qt.expect(qt.jmat(self.spin(), "y"), self.state),\
                         qt.expect(qt.jmat(self.spin(), "z"), self.state)])
        axis = normalize(mink)[1:]
        direction = normalize(axis).tolist()
        length = np.linalg.norm(axis)
        return [direction, length]

    def evolve(self, operator, dt=None, inverse=False):
        if dt == None:
            dt = self.dt
        unitary = (-2*math.pi*1j*operator*dt).expm()
        if inverse:
            unitary = unitary.dag()
        self.state = unitary*self.state

    def rotate(self, pole, inverse=False):
        if pole == "x":
            self.evolve(qt.jmat(self.spin(), "x"), inverse=inverse)
        elif pole == "y":
            self.evolve(qt.jmat(self.spin(), "y"), inverse=inverse)
        elif pole == "z":
            self.evolve(qt.jmat(self.spin(), "z"), inverse=inverse)

    def update(self):
        if self.evolving and self.energy != None:
          self.evolve(self.energy)

    def stars(self):
        return q_SurfaceXYZ(self.state)

    def create_star(self):
        xyz = q_SurfaceXYZ(qt.rand_ket(2))
        self.state = SurfaceXYZ_q(self.stars() + xyz)
        self.energy = qt.rand_herm(self.n())

    def destroy_star(self):
        if self.n() > 2:
            self.state = SurfaceXYZ_q(self.stars()[1:])
            self.energy = qt.rand_herm(self.n())

    def pretty_state(self):
        return np.array_str(self.state.full().T[0], precision=2)