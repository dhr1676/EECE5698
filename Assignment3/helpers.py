# -*- coding: utf-8 -*-
import numpy as np
from SparseVector import SparseVector


def estimateGrad(fun, x, delta):
    """ Given a real-valued function fun, estimate its gradient numerically.

    Inputs are:
       -fun: a function of one variable
       -x: an input, represented as a sparse vector
       -delta: a small real value

        Output is:
       -grad: an estimate of gradient ∇f(x), represented as a sparse vector.
    """
    grad = SparseVector({})
    for key in x:
        e = SparseVector({})
        e[key] = 1.0
        grad[key] = (fun(x + delta * e) - fun(x)) / delta
    return grad
