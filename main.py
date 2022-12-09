import readVTK
from body import Body
import taichi as ti
from material import *
from mesh_data import heartMesh
from PBDQuasiStaticStepping import PBD_with_Continuous_Materials

if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    # filename = "./data/heart1/heart_origin_bou_tag.vtk"
    # vertex, tet, bou_tag = readVTK.read_vtk(filename)

    # 顶点位置
    pos_np = np.array(heartMesh['verts'], dtype=float)
    # 四面体顶点索引
    tet_np = np.array(heartMesh['tetIds'], dtype=int)
    pos_np = pos_np.reshape((-1, 3))
    tet_np = tet_np.reshape((-1, 4))
    # 顶点fiber方向
    fiber_np = np.array(heartMesh['fiberDirection'], dtype=float)
    fiber_np = fiber_np.reshape((-1, 3))
    body = Body(pos_np, tet_np, fiber_np)
    Youngs_Modulus = 1000.
    Poisson_Ratio = 0.49
    # material = Stable_Neo_Hookean(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio)
    material = Stable_Neo_Hookean_with_active(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio, active_tension=60)
    sys = PBD_with_Continuous_Materials(body, material, 10)

    sys.show()


