# import wildmeshing as wm
import pyvista as pv
import tetgen
import meshio
import numpy as np
import vtk

# wm.tetrahedralize("./test1.obj", "./test1/test1_out.msh")
# a = meshio.read("./test1.obj", file_format="obj")
# mesh = pv.Cube(center=(5., 0.5, 0.5), x_length=10., y_length=5., z_length=5.)
# mesh.plot(show_edges=True)

# tet = tetgen.TetGen(mesh)
# tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
# grid = tet.grid
# grid.plot(show_edges=True)

