# -*- coding: utf-8 -*-
# Python packages
import math
import numpy as np
import os
import shutil


""" FOR CLASS ROOM ************************************************************ """
import sys
sys.path.insert(0, "/home/pellegrino/Documents/Pollution/mrgteaching/mrgpylinux")
sys.path.insert(0, "/home/pellegrino/Documents/Pollution/mrgteaching")
from pytransform import pyarmor_runtime
pyarmor_runtime()
""" ******************************************************************************** """


# MRG packages
import femcalc
from femcalc import examples
from femcalc import benchmarks
from femcalc import meshgrid
import femcalc.meshgrid.graphics
import femcalc.benchmarks.meshes
import mrgpylinux


try:
    id_ = sys.argv[1]
except:
    print("You must pass a parameter to identify which image you're using")
    exit(1)

(id_dir, ext) = os.path.splitext(id_)
PACKAGE_DIR = os.path.dirname(os.path.abspath(mrgpylinux.__file__))
#DEMO_DIR = os.path.join(PACKAGE_DIR, 'examples')
INPUT_DATA_DIR = os.path.join(PACKAGE_DIR, '..', 'data', 'input')
OUTPUT_DATA_DIR = os.path.join(PACKAGE_DIR, '..', 'data', 'output', id_dir)
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DATA_DIR, id_dir)
if os.path.exists(OUTPUT_DATA_DIR) and os.path.isdir(OUTPUT_DATA_DIR):
    shutil.rmtree(OUTPUT_DATA_DIR)

os.mkdir(OUTPUT_DATA_DIR)

def normalizedMesh(filename):
    """Translate and scale the mesh.
    """
    mesh = meshgrid.iomrg.readMrgFromImage(filename, cell_size=1.)
    moved_mesh = meshgrid.geometry.anchorMeshAt(mesh, [0.0, 0.0, 0.0],
                                            mode='lbf', inplace=True)
    box = meshgrid.geometry.BoundingBox.fromMrgMesh(moved_mesh)
    max_len = max(box.extents)
    #factor = 1.0 / box.extents[1]
    factor = math.pi / box.extents[1]
    print(f'before box.extents {box.extents}')

    moved_mesh = meshgrid.geometry.scale(moved_mesh, [factor])
    box = meshgrid.geometry.BoundingBox.fromMrgMesh(moved_mesh)
    print(f'after box.extents {box.extents}')
    return moved_mesh


class DemoBoundarySubmeshes(examples.Demo):
    def demonstrate(self):
        filename = os.path.join(INPUT_DATA_DIR, id_)

#        (root, ext) = os.path.splitext(filename)
#        input_file = root + '.png'
#        domain_mesh = meshgrid.iomrg.readMrgFromImage(input_file,cell_size=1/100)
        domain_mesh = normalizedMesh(filename)
        domain_l2g = np.arange(0, domain_mesh.numb_node, 1, dtype='int64')

        output_file = OUTPUT_DATA_FILE
        meshgrid.iomrg.writeVtk(output_file, domain_mesh, id=None)
        meshgrid.iomrg.writeMtx(output_file, domain_l2g, id=None)

        skinner = meshgrid.adjacency.Skinner.fromMesh(domain_mesh)
        boundary_mesh = skinner.buildMesh()
        boundary_l2g = skinner.l2g_nodes()
        output_file = OUTPUT_DATA_FILE +  'boundary'
        meshgrid.iomrg.writeVtk(output_file, boundary_mesh, id=None)
        meshgrid.iomrg.writeMtx(output_file, boundary_l2g, id=None)

        lbf = np.min(boundary_mesh.node_coord,axis=0)
        rtb = np.max(boundary_mesh.node_coord,axis=0)
        eps = 1.E-4  # to replace by computing the number of cells
        south_box = [lbf, [rtb[0]+eps,lbf[1]+eps,0]]
        north_box = [[lbf[0],rtb[1]-eps,0], rtb+eps]
        west_box = [lbf, [lbf[0]+eps,rtb[1]+eps,0]]
        submesher = meshgrid.adjacency.MeshMaskFilter(boundary_mesh,l2g=boundary_l2g)

        south_mask = meshgrid.adjacency.getMeshMaskFromBox(submesher.mesh, south_box)
        submesher.setMask(south_mask)
        south_mesh = submesher.buildMesh()
        south_l2g = submesher.l2g_nodes()
        output_file = OUTPUT_DATA_FILE
        meshgrid.iomrg.writeVtk(output_file, south_mesh, id=0)
        meshgrid.iomrg.writeMtx(output_file, south_l2g, id=0)

        north_mask = meshgrid.adjacency.getMeshMaskFromBox(submesher.mesh, north_box)
        submesher.setMask(north_mask)
        north_mesh = submesher.buildMesh()
        north_l2g = submesher.l2g_nodes()
        output_file = OUTPUT_DATA_FILE
        meshgrid.iomrg.writeVtk(output_file, north_mesh, id=2)
        meshgrid.iomrg.writeMtx(output_file, north_l2g, id=2)

        west_mask = meshgrid.adjacency.getMeshMaskFromBox(submesher.mesh, west_box)
        submesher.setMask(west_mask)
        west_mesh = submesher.buildMesh()
        west_l2g = submesher.l2g_nodes()
        output_file = OUTPUT_DATA_FILE
        meshgrid.iomrg.writeVtk(output_file, west_mesh, id=3)
        meshgrid.iomrg.writeMtx(output_file, west_l2g, id=3)

        submesher.resetMask(value=False)
        submesher.makeUnionMask(south_mask, north_mask, west_mask)
        submesher.negateMask()
        east_mesh = submesher.buildMesh()
        east_l2g = submesher.l2g_nodes()
        output_file = OUTPUT_DATA_FILE
        meshgrid.iomrg.writeVtk(output_file, east_mesh, id=1)
        meshgrid.iomrg.writeMtx(output_file, east_l2g, id=1)

        # viewer = meshgrid.graphics.MeshViewer()
        # viewer.registerMesh(south_mesh, 'south')
        # viewer.registerMesh(east_mesh, 'east')
        # viewer.registerMesh(north_mesh, 'north')
        # viewer.registerMesh(west_mesh, 'west')
        # viewer.show('south', color='red', facecolor='red', linewidth=3)
        # viewer.show('east', color='blue', facecolor='blue', linewidth=3)
        # viewer.show('north', color='green', facecolor='green', linewidth=3)
        # viewer.show('west', color='black', facecolor='black', linewidth=3)
        # viewer.keep()
        return


if __name__ == '__main__':
    demo = DemoBoundarySubmeshes()
    demo.run()
