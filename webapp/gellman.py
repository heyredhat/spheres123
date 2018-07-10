"""Generate generalized Gell-Mann matrices.
  .. module:: gellmann.py
     :synopsis: Generate generalized Gell-Mann matrices
  .. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>
"""
import numpy as np
from itertools import product

def gellmann(j, k, d):
    r"""Returns a generalized Gell-Mann matrix of dimension d.
    
    According to the convention in *Bloch Vectors for Qubits* by Bertlmann and
    Krammer (2008), returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
    :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`, :math:`\Lambda^{jk}_a`
    for :math:`1\leq j<k\leq d`, and :math:`I` for :math:`j=k=d`.
    Parameters
    ----------
    j : positive integer
        Index for generalized Gell-Mann matrix
    k : positive integer
        Index for generalized Gell-Mann matrix
    d : positive integer
        Dimension of the generalized Gell-Mann matrix
    Returns
    -------
    numpy.array
        A genereralized Gell-Mann matrix.
    """

    if j > k:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = 1
        gjkd[k - 1][j - 1] = 1
    elif k > j:
        gjkd = np.zeros((d, d), dtype=np.complex128)
        gjkd[j - 1][k - 1] = -1.j
        gjkd[k - 1][j - 1] = 1.j
    elif j == k and j < d:
        gjkd = np.sqrt(2/(j*(j + 1)))*np.diag([1 + 0.j if n <= j
                                               else (-j + 0.j if n == (j + 1)
                                                     else 0 + 0.j)
                                               for n in range(1, d + 1)])
    else:
        gjkd = np.diag([1 + 0.j for n in range(1, d + 1)])

    return gjkd

def get_basis(d):
    r"""Return a basis of operators.
    
    The basis is made up of orthogonal Hermitian operators on a Hilbert space
    of dimension d, with the identity element in the last place.
    Parameters
    ----------
    d : int
        The dimension of the Hilbert space.
    Returns
    -------
    list of numpy.array
        The basis of operators.
    """
    return [gellmann(j, k, d) for j, k in product(range(1, d + 1), repeat=2)]