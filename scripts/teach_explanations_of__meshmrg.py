# -*- coding: utf-8 -*-
"""This module contains the mesh class.
"""

__authors__ = 'RP, FM'
__copyright__ = '(c) Magoules Research Group (MRG)'
__date__ = '2019.09.13 11:19:46'
__version__ = '0.10.0'

# Python packages
import collections
import enum
import numpy as np
import sys
import sympy

from femcalc.meshgrid.refcells import CellFamily, lambdify_wrapper

class ElemType(enum.IntEnum):
    """Enumeration class for element types supported by MRG Mesh.

    The names and values of enumeration correspond to VTK element
    types.
    >>> ElemType.LINE
    """
    VERTEX = 1
    LINE = 3
    TRIANGLE = 5
    QUAD = 9
    TETRA = 10
    HEXAHEDRON = 12

    @classmethod
    def fromStr(cls, stg):
        """Convert string to enum regardless of string case.

        >>> _mrgmesh.ElemType.FromStr('line')
        >>> _mrgmesh.ElemType.FromStr('LinE')
        """
        ...

    @classmethod
    def convert(cls, items):
        """Convert item to the list of enums.

        >>> a = mesh._mrgmesh.ElemType.convert(('line','quad'))
        >>> b = mesh._mrgmesh.ElemType.convert((3,9))
        >>> c = mesh._mrgmesh.ElemType.convert(a)
        """
        ...

    @property
    def numb_nodes(self):
        ...


class Mesh(object):
    """Mesh class for finite element mesh.
    """

    # init
    def __init__(self):
        """Variables to represent a mesh and data on this mesh.
        """
        # spatial dimensions of nodes
        self.space_dim = 3
        # number of nodes
        self.numb_node = 0
        # 2D array storing for each node its coordinates
        self.node_coord = np.array([])
        # dictionary of node data
        # The array stored in the dictionary should always be 2D arrays, i.e.,
        # its shape has to be of length 2
        self.node_data = {}

        # number of finite elements
        self.numb_elem = 0
        # 1D array storing for each element its nodes
        self.elem2node = np.array([], dtype='int64')
        # 1D array storing for each element the index of its first node in elem2node
        self.p_elem2node = np.array([0], dtype='int64')
        self.elem_geotype = np.array([], dtype='int')
        # dictionary of elem data
        # The array stored in the dictionary should always be 2D arrays, i.e.,
        # its shape has to be of length 2
        self.elem_data = {}

    @property
    def cells_family(self):
        ...

    @property
    def elem_types(self):
        """Return unique sequence of cells' geometric tags
        """
        ...


    def addElemField(self, field_name, data_array, overwrite=False, copy=False):
        """Adds array as element data.
        """
        ...

    def addNodeField(self, field_name, data_array, overwrite=False, copy=False):
        """Adds array as node data.
        """
        ...

    def evalNodeField(self, field_name, expression, overwrite=False, store=True):
        """Evaluate field at nodes and optionally store it in mesh.

        Parameters
        ----------
        field_name : str
            name of the data array to be added in mesh
        expression: str
            string of the symbolic expression to be evaluated.
            (coordinates are denoted by x,y,z)
            (the expression can be passed within or without square bracket)
        overwrite: bool, optional
            If True the existing field with the field_name is silently overwritten.
            If False and the field_name exists raise KeyError exception.
            The default is False.
        store : bool, optional
            If True store evaluated field as node data array.
            If False just evaluate and return it but do not store as node data.
            The default is True.

        Returns
        -------
        numpy.array(dtype=float64)
            2D array of shape (N, C) where N is number of nodes and C is number
            of field components.
        """
        ...

    def writeToStdout(self):
        """Write mesh class on stdout.
        """
        ...

    def countElem(self, selector=None):
        """Return the number of elements in the mesh of type indicated by selector.

        If selector is None all elements are counted.

        Parameters
        ----------
            mesh  - MRG Mesh class
            selector - sequence of ElemType enums, element names or integer
                       values corresponding to VTK element types
        Returns
        -------
            Count of elements of given types
        """
        # count how many items of value in selector are in array
        ...

    def listElemCount(self, selector=None, output=None):
        """Write to output (or sys.stdout) information about number of elements
        in a mesh.

        If selector is given detailed number of elements is written only for
        selected types.
        """
        ...
