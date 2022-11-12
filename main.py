import readVTK
from body import Body
import taichi as ti
from material import *

if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True)
    filename = "./data/heart1/heart_origin_bou_tag.vtk"
    vertex, tet, bou_tag = readVTK.read_vtk(filename)
    body = Body(vertex, tet)
    Youngs_Modulus = 1000.
    Poisson_Ratio = 0.49
    material = NeoHookean(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio)

