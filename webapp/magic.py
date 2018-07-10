import math
import cmath
import mpmath
import sympy
import scipy
import functools
import qutip as qt
import numpy as np
import gellman

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
    zeros = roots.count(float('Inf'))
    roots = [root for root in roots if root != float('Inf')]
    s = sympy.symbols("s")
    polynomial = sympy.Poly(functools.reduce(lambda a, b: a*b, [s-np.conjugate(root) for root in roots]), domain="CC")
    return [complex(0,0)]*zeros + [complex(c) for c in polynomial.coeffs()]

def polynomial_C(polynomial):
    zeros = 0
    for i in range(len(polynomial)):
        if polynomial[i] == 0:
            zeros +=1
        else:
            break
    poles = [float('Inf') for i in range(zeros)]
    roots = [complex(root) for root in mpmath.polyroots(polynomial)]
    return poles+roots

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
    return qt.Qobj(SurfaceXYZ_v(XYZ))
