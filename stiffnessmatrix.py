import taichi as ti
import numpy as np
from body import Body


@ti.data_oriented
class system_of_equations:
    """
    use stiffness matrix and rhs (right hand side) to solve dof (degree of freedom)
    including:
        sparseMtrx: the sparse stiffness matrix utilized to solve the dofs
        rhs: right hand side of the equation system
    """
    def __init__(self, body: Body, material):
        self.body = body
        self.elements, self.vertex = body.elements, body.vertex
        num_vert = self.vertex.shape[0]
        num_tet = self.elements.shape[0]

        # sparseMtrx @ dof = rhs
        self.rhs = ti.field(ti.f32, shape=(body.vertex.shape[0] * 3,))  # right hand side of the equation system
        self.dof = ti.field(ti.f32, shape=(body.vertex.shape[0] * 3,))  # degree of freedom that needs to be solved

        # deformation gradient
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=(num_tet, 1))

        # stress and strain (小变形的无穷小应变和大变形的格林应变)
        self.cauchy_stress = ti.Matrix.field(3, 3, ti.f32, shape=(num_tet, 1))
        self.strain = ti.Matrix.field(3, 3, ti.f32, shape=(num_tet, 1))
        self.mises_stress = ti.field(ti.f32, shape=(num_tet, 1))
        self.ela_psi = ti.field(ti.f32, shape=(num_tet, 1))  # elastic energy density
        self.ela_energy = ti.field(ti.f32, shape=())  # total elastic energy

        # variables related to geometric nonlinear
        self.nodal_force = ti.field(ti.f32, shape=(3 * num_vert))
        self.residual_nodal_force = ti.field(ti.f32, shape=(num_vert * 3, ))

        # dsdx (derivative of shape function with respect to current coordinate), and volume of each guass point
        self.dsdx = ti.Matrix.field(4, 3, ti.f32, shape=(num_tet, 1))
        self.vol = ti.field(ti.f32, shape=(num_tet, 1))

        # constitutive material (e.g., elastic constants) of each element
        self.material = material
        # self.ddsdde = ti.Matrix.field()
