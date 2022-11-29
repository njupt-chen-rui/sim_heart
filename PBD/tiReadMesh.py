from mesh_data import heartMesh
import taichi as ti
import numpy as np
import taichi.math as tm

ti.init(arch=ti.gpu)
# ti.init()

# 四面体顶点数量
numParticles = len(heartMesh['verts']) // 3
# 四面体中所有的边，用于约束四面体的边长
numEdges = len(heartMesh['tetEdgeIds']) // 2
# 四面体数量
numTets = len(heartMesh['tetIds']) // 4
# 渲染使用的四面体表面网格的三角面数量
numSurfs = len(heartMesh['tetSurfaceTriIds']) // 3

# 顶点位置
pos_np = np.array(heartMesh['verts'], dtype=float)
# 四面体顶点索引
tet_np = np.array(heartMesh['tetIds'], dtype=int)
# 边顶点索引
edge_np = np.array(heartMesh['tetEdgeIds'], dtype=int)
# 表面网格顶点索引
surf_np = np.array(heartMesh['tetSurfaceTriIds'], dtype=int)
# 顶点fiber方向
fiber_np = np.array(heartMesh['fiberDirection'], dtype=float)
# 顶点sheet方向
sheet_np = np.array(heartMesh['sheetDirection'], dtype=float)
# 顶点normal方向
normal_np = np.array(heartMesh['normalDirection'], dtype=float)
edge_set_np = np.array(heartMesh['edge_set'], dtype=int)
tet_set_np = np.array(heartMesh['tet_set'], dtype=int)
# bou_tag
bou_tag_np = np.array(heartMesh['bou_tag'], dtype=int)
# contraction
contraction_np = np.array(heartMesh['contraction'], dtype=float)

pos_np = pos_np.reshape((-1, 3))
tet_np = tet_np.reshape((-1, 4))
edge_np = edge_np.reshape((-1, 2))
surf_np = surf_np.reshape((-1, 3))
fiber_np = fiber_np.reshape((-1, 3))
sheet_np = sheet_np.reshape((-1, 3))
normal_np = normal_np.reshape((-1, 3))

pos = ti.Vector.field(3, ti.f32, numParticles)
tet = ti.Vector.field(4, int, numTets)
edge = ti.Vector.field(2, int, numEdges)
surf = ti.Vector.field(3, int, numSurfs)
# 顶点纤维方向
vert_fiber = ti.Vector.field(3, ti.f32, numParticles)
vert_sheet = ti.Vector.field(3, ti.f32, numParticles)
vert_normal = ti.Vector.field(3, ti.f32, numParticles)
# graph color
edge_set = ti.field(ti.int32, numEdges)
tet_set = ti.field(ti.int32, numTets)
# bou_tag
bou_tag = ti.field(ti.int32, numParticles)
# contraction
contraction = ti.field(ti.f32, 50)

# 顶点位置
pos.from_numpy(pos_np)
# 四面体顶点索引
tet.from_numpy(tet_np)
# 边顶点索引
edge.from_numpy(edge_np)
# 表面网格顶点索引
surf.from_numpy(surf_np)
# 顶点纤维方向
vert_fiber.from_numpy(fiber_np)
vert_sheet.from_numpy(sheet_np)
vert_normal.from_numpy(normal_np)
# graph color
edge_set.from_numpy(edge_set_np)
tet_set.from_numpy(tet_set_np)
# bou_tag
bou_tag.from_numpy(bou_tag_np)
# contraction
contraction.from_numpy(contraction_np)


# ---------------------------------------------------------------------------- #
#                      precompute the restLen and restVol                      #
# ---------------------------------------------------------------------------- #

# 四面体初始体积
restVol = ti.field(float, numTets)
restLen = ti.field(float, numEdges)
invMass = ti.field(float, numParticles)
# 形变形状矩阵
Ds = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
# 参考形状矩阵
Dm = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
# Dm^{-1}
invDm = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
# Dm^{-T}
invDmT = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
# 形变梯度
F = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
# C = F^T@F
C = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)

# I_1 = tr(C)
I1 = ti.field(ti.f32, numTets)
I1ref = ti.field(ti.f32, numTets)
# I_4f = f \cdot (C@f)
I4f = ti.field(ti.f32, numTets)
I4fref = ti.field(ti.f32, numTets)
# the 1st Piola-Kirchhoff stress tensor
P = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
# 能量密度函数, 标量
Psi = ti.field(ti.f32, shape=numTets)
# 能量, 标量
E = ti.field(ti.f32, shape=numTets)
# 膜电压
Voltage = ti.field(ti.f32, shape=numTets)
# 主动张力求解所需变量
epsilonV = ti.field(ti.f32, shape=numTets)
Ta = ti.field(ti.f32, shape=numTets)
dTadt = ti.field(ti.f32, shape=numTets)
Psi_a = ti.field(ti.f32, shape=numTets)
P_a = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
grad = ti.Vector.field(3, ti.f32, shape=numParticles)

# 能量对位置x的梯度
dEdx012 = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
dEdx = ti.Vector.field(3, ti.f32, shape=(numTets, 4))


# 四面体中fiber方向单位向量
f0 = ti.Vector.field(3, dtype=ti.f32, shape=numTets)
f = ti.Vector.field(3, dtype=ti.f32, shape=numTets)
# 四面体中sheet方向单位向量
s0 = ti.Vector.field(3, dtype=ti.f32, shape=numTets)
s = ti.Vector.field(3, dtype=ti.f32, shape=numTets)

# lambda
para_lambda = ti.field(ti.f32, numTets)
diag1_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
diag1 = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=())
diag1.from_numpy(diag1_np)


@ti.func
# 求解四面体体积
def tetVolume(i):
    id = tm.ivec4(-1, -1, -1, -1)
    for j in ti.static(range(4)):
        id[j] = tet[i][j]
    temp = (pos[id[1]] - pos[id[0]]).cross(pos[id[2]] - pos[id[0]])
    res = temp.dot(pos[id[3]] - pos[id[0]])
    res *= 1.0/6.0
    return res


@ti.kernel
def init_physics():
    # 初始化四面体体积
    for i in restVol:
        restVol[i] = tetVolume(i)
    # 初始化边长
    for i in restLen:
        restLen[i] = (pos[edge[i][0]] - pos[edge[i][1]]).norm()


@ti.kernel
def init_invMass():
    for i in range(numTets):
        pInvMass = 0.0
        if restVol[i] > 0.0:
            pInvMass = 1.0 / (restVol[i] / 4.0)
        for j in ti.static(range(4)):
            invMass[tet[i][j]] += pInvMass


@ti.kernel
def init_Dm():
    for i in range(numTets):
        Dm[i][0, 0] = pos[tet[i][0]][0] - pos[tet[i][3]][0]
        Dm[i][1, 0] = pos[tet[i][0]][1] - pos[tet[i][3]][1]
        Dm[i][2, 0] = pos[tet[i][0]][2] - pos[tet[i][3]][2]
        Dm[i][0, 1] = pos[tet[i][1]][0] - pos[tet[i][3]][0]
        Dm[i][1, 1] = pos[tet[i][1]][1] - pos[tet[i][3]][1]
        Dm[i][2, 1] = pos[tet[i][1]][2] - pos[tet[i][3]][2]
        Dm[i][0, 2] = pos[tet[i][2]][0] - pos[tet[i][3]][0]
        Dm[i][1, 2] = pos[tet[i][2]][1] - pos[tet[i][3]][1]
        Dm[i][2, 2] = pos[tet[i][2]][2] - pos[tet[i][3]][2]


@ti.kernel
def init_invDm():
    for i in range(numTets):
        invDm[i] = Dm[i].inverse()


@ti.kernel
def init_invDmT():
    for i in range(numTets):
        invDmT[i] = invDm[i].transpose()


@ti.kernel
def init_tetFiber():
    for i in range(numTets):
        f[i] = vert_fiber[tet[i][0]] + vert_fiber[tet[i][1]] + vert_fiber[tet[i][2]] + vert_fiber[tet[i][3]]
        f[i] /= 4.0
        f[i] /= tm.length(f[i])
        f0[i] = f[i]
        s[i] = vert_sheet[tet[i][0]] + vert_sheet[tet[i][1]] + vert_sheet[tet[i][2]] + vert_sheet[tet[i][3]]
        s[i] /= 4.0
        s[i] = s[i] - (s[i].dot(f[i])) / (f[i].dot(f[i])) * f[i]
        s[i] /= tm.length(s[i])
        s0[i] = s[i]


@ti.kernel
def init_Ta():
    for i in range(numTets):
        Ta[i] = 0.0


init_physics()
init_invMass()
init_Dm()
init_invDm()
init_invDmT()
init_tetFiber()
init_Ta()
