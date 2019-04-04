import numpy as np


class SparseVector(dict):
    """
       A class implementing a sparse vector as a dictionary. 
       A sparse vector is a vector that contains only a few non-zero values. 
       Each non-zero value is associated with a feature (e.g., "age", "height", "weight"). 
       The sparse vector is represented as a dictionary with features as keys and the
       coordinates of the vector as values. If a feature is not present in the dictionary, 
       it is assumed that the corresponding value of the vector is zero.
    """

    def safeAccess(self, key):
        """ If key is in the dictionary x, return x[key]; otherwise return 0.0.
        """
        if key in self:
            return self[key]
        else:
            return 0.0

    def dot(self, other):
        """ Return the inner product between two sparse vectors represented as dictionaries. That is, given
        sparse vectors x,y, x.dot(y) returns the scalar

        <x,y>

        """
        return sum([self[key] * other[key] for key in self if key in other])

    def __add__(self, other):
        """ Add two sparse vectors represented as dictionaries. That is, given sparse vectors x and y,
            return the sparse vector representing:
                x + y
        """
        l = [(key, self[key] + other[key]) for key in self if key in other]
        l += [(key, self[key]) for key in self if key not in other]
        l += [(key, other[key]) for key in other if key not in self]
        return SparseVector(l)

    def __sub__(self, other):
        """ Subtract two sparse vectors represented as dictionaries. That is, given sparse vectors x and y,
            return the sparse vector representing:
        x - y
        """
        l = [(key, self[key] - other[key]) for key in self if key in other]
        l += [(key, self[key]) for key in self if key not in other]
        l += [(key, -other[key]) for key in other if key not in self]
        return SparseVector(l)

    def __mul__(self, s):
        """ Multiply a sparse vector x with a scalar. That is
                 x * s
        will return a sparse vector containing x's coordinates multiplied by s
        """
        return SparseVector([(key, s * self[key]) for key in self])

    def __rmul__(self, s):
        """ Multiply a sparse vector x with a scalar from the right. That is
        s * x
        will return a sparse vector containint x'y coordinates multiplied by s

        """
        return self * s
