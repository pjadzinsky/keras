# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations, regularizers, constraints
from ..utils.theano_utils import shared_zeros, floatX
from ..utils.generic_utils import make_tuple
from ..regularizers import ActivityRegularizer, Regularizer
from ..layers.core import Layer

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip
import pdb

class Rotate1D(Layer):
    '''
        Rotate representation.
        Can't be used as first layer in a model (no fixed input!)
        Tensor is supposed to be 3D
        First dimension is assumed to be nb_samples.
        Second dimension is assumed to be time.
        Third dimension is the representation at each time.
    '''
    def __init__(self, n):
        super(Rotate1D, self).__init__()
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        y = T.zeros_like(x)
        y = T.set_subtensor(y[:, :l, :], x[:, -l:, :])
        y = T.set_subtensor(y[:, l:, :], x[:, :-l, :])
        return y

    def get_config(self):
        return {"name": self.__class__.__name__,
                "dims": self.dims}

class Rotate(Layer):
    '''
        Rotate a tensor along a given dimension dim > 0
        Tensor can have any number of dimensions but first
        dimension is reserved for nb_samples and can't be rotated
    '''
    def __init__(self, n, dim):
        super(Rotate, self).__init__()
        self.n = n
        self.dim = dim

    def get_output(self, train):
        n = self.n
        dim = self.dim

        N = x.shape[dim]
        total_dim = x.total_dim

        front_slices = (slice(None),)*dim
        back_slices = (slice(None),)*(total_dim-n-1)

        y = T.zeros_like(x)
        y = T.set_subtensor(
                y[front_slices + (slice(0, n),) + back_slices],
                x[front_slices + (slice(N-n,N),) + back_slices])
        y = T.set_subtensor(
                y[front_slices + (slice(n,N),) + back_slices],
                x[front_slices + (slice(0,N-n),) + back_slices])
        return y

    def get_config(self):
        return {"name": self.__class__.__name__,
                "dims": self.dims}

class Pad_Front(Layer):
    def __init__(self, n, dim):
        super(Pad_Front, self).__init__()
        self.n = n
        self.dim = dim
        self.input = T.tensor4()

    def get_output(self, train):
        n = self.n
        dim = self.dim

        X = self.get_input(train)
        in_shape = X.shape
        total_dim = len(in_shape)

        out_shape = in_shape
        out_shape[dim] += n
        out = T.zeros(out_shape)

        front_slices = (slice(None),)*dim
        back_slices = (slice(None),)*(total_dim-n-1)

        indices = front_slices + (slice(n, out_shape[dim]), ) + back_slices
        return T.set_subtensor(out[indices], X)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "pad": self.pad}

class Pad_End(Layer):
    def __init__(self, n, dim):
        super(Pad_Front, self).__init__()
        self.n = n
        self.dim = dim
        self.input = T.tensor4()

    def get_output(self, train):
        n = self.n
        dim = self.dim

        X = self.get_input(train)
        in_shape = X.shape
        total_dim = len(in_shape)

        out_shape = in_shape
        out_shape[dim] += n
        out = T.zeros(out_shape)

        front_slices = (slice(None),)*dim
        back_slices = (slice(None),)*(total_dim-n-1)

        indices = front_slices + (slice(0, out_shape[dim]-n), ) + back_slices
        return T.set_subtensor(out[indices], X)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "pad": self.pad}
