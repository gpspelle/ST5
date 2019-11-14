# -*- coding: utf-8 -*-
# Python packages
import numpy as np


""" FOR CLASS ROOM ************************************************************ """
import sys
sys.path.insert(0, "C:\\Users\\gps_0\\OneDrive\\Documentos\\CS\\Pollution Acoustique\\mrgteaching\\mrgpy")
from pytransform import pyarmor_runtime
pyarmor_runtime()
""" ******************************************************************************** """


# MRG packages
from femcalc import meshgrid
import femcalc.meshgrid.graphics


def simpleMesh():
    """
                      6
                    / |
                  /   |
                 /  2 |
       3 ------ 2 --- 5
       |        |     |
       |   0    |  1  |
       |        |     |
       0 ------ 1 --- 4

    """
    out_mesh = meshgrid._meshmrg.Mesh()

    out_mesh.space_dim = 3
    out_mesh.numb_node = 7
    out_mesh.node_coord = np.empty((out_mesh.numb_node, out_mesh.space_dim))

    out_mesh.numb_elem = 3
    out_mesh.elem2node = np.array([0, 1, 2, 3,
                                   1, 4, 5, 2,
                                   2, 5, 6],
                                  dtype='int64')
    out_mesh.p_elem2node = np.array([0, 4, 8, 11], dtype='int64')
    out_mesh.elem_geotype = np.array([meshgrid._meshmrg.ElemType.QUAD,
                                      meshgrid._meshmrg.ElemType.QUAD,
                                      meshgrid._meshmrg.ElemType.TRIANGLE],
                                     dtype='int64')

    out_mesh.node_coord = np.array([[0., 0., 0.],
                                    [1., 0., 0.],
                                    [1., 1., 0.],
                                    [0., 1., 0.],
                                    [2., 0., 0.],
                                    [2., 1., 0.],
                                    [2., 2., 0.],
                                    ], dtype='float64')
    # local to global numbering
    out_mesh_l2g = np.arange(0, out_mesh.numb_node, 1, dtype='int64')
    return out_mesh, out_mesh_l2g


if __name__ == '__main__':
    mesh, l2g = simpleMesh()
    mesh.writeToStdout()
    field_expression = 'x**2+y**2'
    mesh.evalNodeField('ACOUSTIC_PRESSURE',field_expression)
    viewer = meshgrid.graphics.MeshViewer()
    viewer.registerMesh(mesh, 'yourMeshName')
    viewer.show('yourMeshName',edgecolor='black', facecolor='none')
    viewer.showNodeLabels('yourMeshName')
    viewer.showCellLabels('yourMeshName')
    viewer.keep()
    viewer.showContourf('yourMeshName','ACOUSTIC_PRESSURE',100)
    viewer.keep()