# -*- coding: utf-8 -*-

"""
morphsnakes
===========

This is a Python implementation of the algorithms introduced in the paper

  Márquez-Neila, P., Baumela, L., Álvarez, L., "A morphological approach
  to curvature-based evolution of curves and surfaces". IEEE Transactions
  on Pattern Analysis and Machine Intelligence (PAMI), 2013.

This implementation is intended to be as brief, understandable and self-contained
as possible. It does not include any enhancement to make it fast or efficient.

Any practical implementation of this algorithm should work only over the
neighbor pixels of the 0.5-levelset, not over all the embedding function,
and perhaps should feature multi-threading or GPU capabilities.

The classes MorphGAC and MorphACWE provide most of the functionality of this
module. They implement the Morphological Geodesic Active Contours and the
Morphological Active Contours without Edges, respectively. See the
aforementioned paper for full details.

See test.py for examples of usage.
"""

__author__ = "P. Márquez Neila <p.mneila@upm.es>"

from itertools import cycle

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, \
                        gaussian_filter, gaussian_gradient_magnitude

class fcycle(object):
    
    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)
    
    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)
    

# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0,1,0]]*3), np.flipud(np.eye(3)), np.rot90([[0,1,0]]*3)]
_P3 = [np.zeros((3,3,3)) for i in range(9)]

_P3[0][:,:,1] = 1
_P3[1][:,1,:] = 1
_P3[2][1,:,:] = 1
_P3[3][:,[0,1,2],[0,1,2]] = 1
_P3[4][:,[0,1,2],[2,1,0]] = 1
_P3[5][[0,1,2],:,[0,1,2]] = 1
_P3[6][[0,1,2],:,[2,1,0]] = 1
_P3[7][[0,1,2],[0,1,2],:] = 1
_P3[8][[0,1,2],[2,1,0],:] = 1

_aux = np.zeros((0))
def SI(u):
    """SI operator.
    Passando o levelset (matriz com 0 fora do contorno (inactive pixels) e 1 dentro do contorno (active pixels))
    """
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")
    
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for _aux_i, P_i in zip(_aux, P):
        _aux_i[:] = binary_erosion(u, P_i)
    
    return _aux.max(0)

def IS(u):
    """IS operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")
    
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for _aux_i, P_i in zip(_aux, P):
        _aux_i[:] = binary_dilation(u, P_i)
    
    return _aux.min(0)

# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])

# Stopping factors (function g(I) in the paper). 
# EQ1 (MORPHOLOGICAL SNAKES)
def gborders(img, alpha=1.0, sigma=1.0):
    """Stopping criterion for image borders."""
    # The norm of the gradient. |gradG(I) * Imagem|
    gradnorm = gaussian_gradient_magnitude(img, sigma, mode='constant')
    return 1.0/np.sqrt(1.0 + alpha*gradnorm)

#EQ2 (MORPHOLOGICAL SNAKES)
def glines(img, sigma=1.0):
    """Stopping criterion for image black lines."""
    return gaussian_filter(img, sigma)

class MorphACWE(object):
    """Morphological ACWE based on the Chan-Vese energy functional."""
    
    def __init__(self, data, smoothing=1, lambda1=1, lambda2=1):
        """Create a Morphological ACWE solver.
        
        Parameters
        ----------
        data : ndarray
            The image data.
        smoothing : scalar
            The number of repetitions of the smoothing step (the
            curv operator) in each iteration. In other terms,
            this is the strength of the smoothing. This is the
            parameter µ. -> baixo para obj pequenos e alto para obj grandes
        lambda1, lambda2 : scalars
            Relative importance of the inside pixels (lambda1)
            against the outside pixels (lambda2).
        """
        self._u = None
        self.smoothing = smoothing
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.data = data
    
    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0
    
    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    
    def step(self):
        """Perform a single step of the morphological Chan-Vese evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u
        
        if u is None:
            raise ValueError("the levelset function is not set (use set_levelset)")
        
        data = self.data

        # nao precisa do passo 1 (?)
        
        # Determine c0 and c1.
        inside = u>0
        outside = u<=0
        c0 = data[outside].sum() / float(outside.sum())
        c1 = data[inside].sum() / float(inside.sum())
        
        # Image attachment.
        dres = np.array(np.gradient(u))
        abs_dres = np.abs(dres).sum(0)
        #aux = abs_dres * (c0 - c1) * (c0 + c1 - 2*data)
        # evolucao da curva EQ 32 morphological approach...
        # segundo passo
        aux = abs_dres * (self.lambda1*(data - c1)**2 - self.lambda2*(data - c0)**2)
        
        res = np.copy(u)
        res[aux < 0] = 1
        res[aux > 0] = 0
        
        # Smoothing.
        #terceiro passo EQ 32
        for i in range(self.smoothing):
            res = curvop(res)
        
        self._u = res
    
    def run(self, iterations):
        """Run several iterations of the morphological Chan-Vese method."""
        for i in range(iterations):
            self.step()
    

class MorphGAC(object):
    """Morphological GAC based on the Geodesic Active Contours."""
    
    def __init__(self, data, smoothing=1, threshold=0, balloon=0):
        """Create a Morphological GAC solver.
        
        Parameters
        ----------
        data : array-like
            The stopping criterion g(I). See functions gborders and glines.
        smoothing : scalar
            The number of repetitions of the smoothing step in each
            iteration. This is the parameter µ. -> baixo para obj pequenos e alto para obj grandes
        threshold : scalar
            The threshold that determines which areas are affected
            by the morphological balloon. This is the parameter θ.
        balloon : scalar
            The strength of the morphological balloon. This is the parameter ν.
        """
        self._u = None #contorno C => levelset
        self._v = balloon #direção da curva => p/ dentro ou p/ fora
        self._theta = threshold #enquanto g(I) for maior, usa força do balão e movimenta curva, se for menor não movimenta a curva
        self.smoothing = smoothing
        self.name = "Morphological GAC"
        
        self.set_data(data)
    
    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1 #pontos dentro da curva => objeto segmentado
        self._u[u<=0] = 0 #pontos fora da curva => background
    
    def set_balloon(self, v):
        self._v = v
        self._update_mask()
    
    def set_threshold(self, theta):
        self._theta = theta
        self._update_mask()
    
    def set_data(self, data):
        self._data = data
        self._ddata = np.gradient(data)
        self._update_mask()
        # The structure element for binary dilation and erosion.
        self.structure = np.ones((3,)*np.ndim(data))
    
    def _update_mask(self):
        """Pre-compute masks for speed."""
        self._threshold_mask = self._data > self._theta
        self._threshold_mask_v = self._data > self._theta/np.abs(self._v)
    
    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    data = property(lambda self: self._data,
                        set_data,
                        doc="The data that controls the snake evolution (the image or g(I)).")
    balloon = property(lambda self: self._v,
                        set_balloon,
                        doc="The morphological balloon parameter (ν (nu, not v)).")
    threshold = property(lambda self: self._theta,
                        set_threshold,
                        doc="The threshold value (θ).")
    
    def step(self):
        """Perform a single step of the morphological snake evolution."""
        # Assign attributes to local variables for convenience.
        u = self._u
        gI = self._data
        dgI = self._ddata #gradiente
        theta = self._theta
        v = self._v
        
        if u is None:
            raise ValueError("the levelset is not set (use set_levelset)")
        
        res = np.copy(u) #contorno atual
        
        # Balloon.
        #movimenta o balão
        if v > 0:
            aux = binary_dilation(u, self.structure) #sup D_h
        elif v < 0:
            aux = binary_erosion(u, self.structure) #inf E_h
        if v!= 0:
            res[self._threshold_mask_v] = aux[self._threshold_mask_v]
        
        # Image attachment.
        aux = np.zeros_like(res)
        #faz o gradiente e move a evolui o levelset
        dres = np.gradient(res)
        for el1, el2 in zip(dgI, dres):
            aux += el1*el2
        res[aux > 0] = 1
        res[aux < 0] = 0
        
        # Smoothing.
        #suaviza a curva nas bordas
        for i in range(self.smoothing):
            res = curvop(res)
        #atualiza curva atual
        self._u = res
    
    def run(self, iterations):
        """Run several iterations of the morphological snakes method."""
        for i in range(iterations):
            self.step()
