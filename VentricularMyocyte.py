"""
    Single cell electrophysiological model
"""
import taichi as ti
import numpy as np
import taichi.math as tm


@ti.dataclass
class TP_Variables:
    # voltage + backup
    Volt: float
    Volt2: float
    Cai: float
    CaSR: float
    CaSS: float
    Nai: float
    Ki: float

    # states of voltage and time dependent gates
    # INa
    M: float
    H: float
    J: float
    # IKr
    # IKr1
    Xr1: float
    # IKr2
    Xr2: float
    # IKs
    Xs: float
    # Ito1
    R: float
    S: float
    # ICa
    D: float
    F: float
    F2: float
    FCass: float
    # Irel
    RR: float
    OO: float
    # total current
    Itot: float

    @ti.kernel
    def init(self, V_init, Cai_init, CaSR_init, CaSS_init, Nai_init, Ki_init):
        Volt = V_init
        Volt2 = V_init
        Cai = Cai_init
        CaSR = CaSR_init
        CaSS = CaSS_init
        Nai = Nai_init
        Ki = Ki_init
        M = 0.
        H = 0.75
        J = 0.75
        Xr1 = 0.
        Xr2 = 1.
        Xs = 0.
        R = 0.
        S = 1.
        D = 0.
        F = 1.
        F2 = 1.
        FCass = 1.
        RR = 1.
        OO = 0.


@ti.data_oriented
class TP:
    def __init__(self, vertex: np.ndarray, elements: np.ndarray, vert_fiber: np.ndarray) -> None:
        # len(vertex[0]) = 3, len(vertex) = num_vert
        self.num_vertex = len(vertex)
        self.vertex = ti.Vector.field(3, dtype=ti.f32, shape=(self.num_vertex, ))
        self.vertex.from_numpy(vertex)
        # len(elements[0]) = 4, len(elements) = num_tet
        self.num_tet = len(elements)
        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=(self.num_tet, ))
        self.elements.from_numpy(elements)
        self.np_vertex = vertex
        self.np_elements = elements





if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True)
    filename = "./data/heart1/heart_origin_bou_tag.vtk"
    points, mesh, bou_tag = readVTK.read_vtk(filename)
    body = Body(vertex=points, elements=mesh)

    body.show()
