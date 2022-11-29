import taichi as ti
import numpy as np
from body import Body
import readVTK
from material import *


class PBDQuasiStaticStepping:
    def __init__(self, body: Body, material):
        self.body = body
        self.material = material
        self.vertex = body.vertex
        self.elements = body.elements
        self.DmInv = body.DmInv
        self.Ds = ti.Matrix.field(3, 3, ti.f32, shape=(body.num_tet,))
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=(body.num_tet,))

    @ti.kernel
    def Compute_P_Psi(self):
        num_ele = self.elements.shape[0]
        for i in range(num_ele):
            self.Ds[i][0, 0] = self.vertex[self.elements[i][0]][0] - self.vertex[self.elements[i][3]][0]
            self.Ds[i][1, 0] = self.vertex[self.elements[i][0]][1] - self.vertex[self.elements[i][3]][1]
            self.Ds[i][2, 0] = self.vertex[self.elements[i][0]][2] - self.vertex[self.elements[i][3]][2]
            self.Ds[i][0, 1] = self.vertex[self.elements[i][1]][0] - self.vertex[self.elements[i][3]][0]
            self.Ds[i][1, 1] = self.vertex[self.elements[i][1]][1] - self.vertex[self.elements[i][3]][1]
            self.Ds[i][2, 1] = self.vertex[self.elements[i][1]][2] - self.vertex[self.elements[i][3]][2]
            self.Ds[i][0, 2] = self.vertex[self.elements[i][2]][0] - self.vertex[self.elements[i][3]][0]
            self.Ds[i][1, 2] = self.vertex[self.elements[i][2]][1] - self.vertex[self.elements[i][3]][1]
            self.Ds[i][2, 2] = self.vertex[self.elements[i][2]][2] - self.vertex[self.elements[i][3]][2]

        for i in range(num_ele):
            self.F[i] = self.Ds[i] @ self.DmInv[i]



if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    filename = "./data/heart1/heart_origin_bou_tag.vtk"
    vertex, tet, bou_tag = readVTK.read_vtk(filename)
    body = Body(vertex, tet)
    Youngs_Modulus = 1000.
    Poisson_Ratio = 0.49
    material = NeoHookean(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio)
    sys = PBDQuasiStaticStepping(body, material)
