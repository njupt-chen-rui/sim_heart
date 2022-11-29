import tiReadMesh
from tiReadMesh import *
import math

frame = ti.field(ti.int32, shape=())
numSubsteps = 10
m = 1
dt = 1.0 / 60.0 / numSubsteps
# dt = 1.0 / 30.0 / numSubsteps
edgeCompliance = 100.0
volumeCompliance = 0.0
max_edge_set = 23
max_tet_set = 36

prevPos = ti.Vector.field(3, float, numParticles)
vel = ti.Vector.field(3, float, numParticles)

surf_show = ti.field(int, numSurfs * 3)
surf_show.from_numpy(surf_np.flatten())
Strain = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())
# 收缩率，每个四面体一个[0,1)的float值
actval = ti.field(ti.f32, shape=numTets)
act = ti.Vector.field(3, float, numTets)
Act = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
ActInv = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
FAct = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
# 不变量
Is = ti.Vector.field(3, float, numTets)
gs = ti.Vector.field(9, float, shape=(numTets, 3))
iso_psi = ti.field(ti.f32, shape=numTets)
aniso_psi = ti.field(ti.f32, shape=numTets)
# dFactdF = ti.Matrix.field(n=9, m=9, dtype=ti.f32, shape=numTets)
dFactdF = ti.field(float, shape=(numTets, 9, 9))
dFactdFT = ti.field(float, shape=(numTets, 9, 9))
iso_dpsi_vec = ti.Vector.field(9, float, numTets)
aniso_dpsi_vec = ti.Vector.field(9, float, numTets)
dpsi_vec = ti.Vector.field(9, float, numTets)
iso_dpsi = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
aniso_dpsi = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=numTets)
ga = ti.Vector.field(9, float, shape=numTets)
fiber_strength = 20

init_pos = ti.Vector.field(3, ti.f32, numParticles)

@ti.kernel
# 施加外力
def preSolve():
    g = tm.vec3(0, 0, 0)
    # g = tm.vec3(0, -9.8, 0)
    for i in pos:
        prevPos[i] = pos[i]
        vel[i] += g * dt
        pos[i] += vel[i] * dt
        if pos[i].y < 0.0:
            pos[i] = prevPos[i]
            pos[i].y = 0.0


# 解算约束
def solve():
    for i in range(1, max_tet_set + 1):
        solveFEM(i)

    # for i in range(1, max_edge_set + 1):
    #     solveEdge(i)

    for i in range(1, max_tet_set + 1):
        solveVolume(i)

    solve_bound()


@ti.kernel
def solveEdge(edge_set_id: ti.int32):
    alpha = edgeCompliance / dt / dt
    grads = tm.vec3(0, 0, 0)
    for i in range(numEdges):
        if edge_set[i] == edge_set_id:
            id0 = edge[i][0]
            id1 = edge[i][1]

            grads = pos[id0] - pos[id1]
            Len = grads.norm()
            grads = grads / Len
            C = Len - restLen[i]
            w = invMass[id0] + invMass[id1]
            s = -C / (w + alpha)

            pos[id0] += grads * s * invMass[id0]
            pos[id1] += grads * (-s * invMass[id1])


@ti.kernel
def solveVolume(tet_set_id: ti.int32):
    alpha = volumeCompliance / dt / dt
    grads = [tm.vec3(0, 0, 0), tm.vec3(0, 0, 0), tm.vec3(0, 0, 0), tm.vec3(0, 0, 0)]

    for i in range(numTets):
        if tet_set[i] == tet_set_id:
            id = tm.ivec4(-1, -1, -1, -1)
            for j in ti.static(range(4)):
                id[j] = tet[i][j]
            grads[0] = (pos[id[3]] - pos[id[1]]).cross(pos[id[2]] - pos[id[1]])
            grads[1] = (pos[id[2]] - pos[id[0]]).cross(pos[id[3]] - pos[id[0]])
            grads[2] = (pos[id[3]] - pos[id[0]]).cross(pos[id[1]] - pos[id[0]])
            grads[3] = (pos[id[1]] - pos[id[0]]).cross(pos[id[2]] - pos[id[0]])

            w = 0.0
            for j in ti.static(range(4)):
                w += invMass[id[j]] * (grads[j].norm()) ** 2

            vol = tetVolume(i)
            C = (vol - restVol[i]) * 6.0
            s = -C / (w + alpha)

            for j in ti.static(range(4)):
                pos[tet[i][j]] += grads[j] * s * invMass[id[j]]


@ti.func
def getAct(i: ti.int32):
    a = 1.0 - actval[i]
    # act[i] = tm.vec3(1, ti.sqrt(1.0 / a), ti.sqrt(1.0 / a))
    act[i] = tm.vec3(a, ti.sqrt(1.0 / a), ti.sqrt(1.0 / a))
    dir0 = f[i]
    tmp = tm.vec3(1., 0., 0.)
    dir1 = dir0.cross(tmp)
    if tm.length(dir1) < 1e-3:
        tmp = tm.vec3(0., 1., 0.)
        dir1 = dir0.cross(tmp)
    dir1 = dir1 / tm.length(dir1)
    dir2 = dir0.cross(dir1)
    dir = tm.mat3([dir0, dir1, dir2])

    R = tm.mat3([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            R[j, k] = dir[k, j]

    Act[i] = tm.mat3([(0, 0, 0), (0, 0, 0), (0, 0, 0)])
    for j in ti.static(range(3)):
        Act[i][j, j] = act[i][j]
    Act[i] = R @ Act[i] @ R.transpose()
    ActInv[i] = Act[i].inverse()


@ti.func
def ComputePsiDeriv(i: ti.int32):
    F[i] = Ds[i] @ invDm[i]
    FAct[i] = F[i] @ ActInv[i]
    paraE = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = paraE * nu / ((1 + nu) * (1 - 2 * nu))
    LameMu[None] = paraE / (2 * (1 + nu))
    U, sigma, V = ti.svd(FAct[i], ti.f32)
    if sigma[2, 2] < 0:
        sigma[2, 2] = -sigma[2, 2]
    Is[i][0] = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
    Is[i][1] = sigma[0, 0] * sigma[0, 0] + sigma[1, 1] * sigma[1, 1] + sigma[2, 2] * sigma[2, 2]
    Is[i][2] = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]
    R = U @ V.transpose()
    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            gs[i, 0][j * 3 + k] = R[j, k]
    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            gs[i, 1][j * 3 + k] = 2. * FAct[i][j, k]
    Jcol0 = tm.vec3(FAct[i][1, 1] * FAct[i][2, 2] - FAct[i][2, 1] * FAct[i][1, 2],
                    FAct[i][2, 1] * FAct[i][0, 2] - FAct[i][0, 1] * FAct[i][2, 2],
                    FAct[i][0, 1] * FAct[i][1, 2] - FAct[i][1, 1] * FAct[i][0, 2])
    Jcol1 = tm.vec3(FAct[i][1, 2] * FAct[i][2, 0] - FAct[i][2, 2] * FAct[i][1, 0],
                    FAct[i][2, 2] * FAct[i][0, 0] - FAct[i][0, 2] * FAct[i][2, 0],
                    FAct[i][0, 2] * FAct[i][1, 0] - FAct[i][1, 2] * FAct[i][0, 0])
    Jcol2 = tm.vec3(FAct[i][1, 0] * FAct[i][2, 1] - FAct[i][2, 0] * FAct[i][1, 1],
                    FAct[i][2, 0] * FAct[i][0, 1] - FAct[i][0, 0] * FAct[i][2, 1],
                    FAct[i][0, 0] * FAct[i][1, 1] - FAct[i][1, 0] * FAct[i][0, 1])
    J = tm.mat3([Jcol0, Jcol1, Jcol2])
    J = J.transpose()
    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            gs[i, 2][j * 3 + k] = J[j, k]
    iso_psi[i] = LameMu[None] / 2. * (Is[i][1] - 3) - LameMu[None] * (Is[i][2] - 1) + LameLa[None] / 2 * (Is[i][2] - 1) * (
                Is[i][2] - 1)
    dphi = tm.vec3(0., 0., 0.)
    dphi[1] = LameMu[None] / 2
    dphi[2] = -LameMu[None] + LameLa[None] * (Is[i][2] - 1)

    # dFactdF[i][0, 0] = dFactdF[i][1, 1] = dFactdF[i][2, 2] = ActInv[i][0, 0]
    # dFactdF[i][3, 0] = dFactdF[i][4, 1] = dFactdF[i][5, 2] = ActInv[i][0, 1]
    # dFactdF[i][6, 0] = dFactdF[i][7, 1] = dFactdF[i][8, 2] = ActInv[i][0, 2]
    #
    # dFactdF[i][0, 3] = dFactdF[i][1, 4] = dFactdF[i][2, 5] = ActInv[i][1, 0]
    # dFactdF[i][3, 3] = dFactdF[i][4, 4] = dFactdF[i][5, 5] = ActInv[i][1, 1]
    # dFactdF[i][6, 3] = dFactdF[i][5, 5] = dFactdF[i][8, 5] = ActInv[i][1, 2]
    #
    # dFactdF[i][0, 6] = dFactdF[i][1, 7] = dFactdF[i][2, 8] = ActInv[i][2, 0]
    # dFactdF[i][3, 6] = dFactdF[i][4, 7] = dFactdF[i][5, 8] = ActInv[i][2, 1]
    # dFactdF[i][6, 6] = dFactdF[i][7, 7] = dFactdF[i][8, 8] = ActInv[i][2, 2]

    dFactdF[i, 0, 0] = dFactdF[i, 1, 1] = dFactdF[i, 2, 2] = ActInv[i][0, 0]
    dFactdF[i, 3, 0] = dFactdF[i, 4, 1] = dFactdF[i, 5, 2] = ActInv[i][0, 1]
    dFactdF[i, 6, 0] = dFactdF[i, 7, 1] = dFactdF[i, 8, 2] = ActInv[i][0, 2]

    dFactdF[i, 0, 3] = dFactdF[i, 1, 4] = dFactdF[i, 2, 5] = ActInv[i][1, 0]
    dFactdF[i, 3, 3] = dFactdF[i, 4, 4] = dFactdF[i, 5, 5] = ActInv[i][1, 1]
    dFactdF[i, 6, 3] = dFactdF[i, 5, 5] = dFactdF[i, 8, 5] = ActInv[i][1, 2]

    dFactdF[i, 0, 6] = dFactdF[i, 1, 7] = dFactdF[i, 2, 8] = ActInv[i][2, 0]
    dFactdF[i, 3, 6] = dFactdF[i, 4, 7] = dFactdF[i, 5, 8] = ActInv[i][2, 1]
    dFactdF[i, 6, 6] = dFactdF[i, 7, 7] = dFactdF[i, 8, 8] = ActInv[i][2, 2]

    for j in ti.static(range(9)):
        for k in ti.static(range(9)):
            dFactdFT[i, j, k] = dFactdF[i, k, j]

    iso_dpsi_vec[i] = dphi[0] * gs[i, 0] + dphi[1] * gs[i, 1] + dphi[2] * gs[i, 2]
    # iso_dpsi_vec[i] = dFactdF[i].transpose() @ iso_dpsi_vec[i]
    # tmp_vec = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    tmp_vec = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for j in ti.static(range(9)):
        for k in ti.static(range(9)):
            tmp_vec[j] += dFactdFT[i, j, k] * iso_dpsi_vec[i][k]
    for j in ti.static(range(9)):
        iso_dpsi_vec[i][j] = tmp_vec[j]

    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            iso_dpsi[i][j, k] = iso_dpsi_vec[i][j * 3 + k]

    # f[i] = FAct[i] @ f0[i]
    # f[i] /= tm.length(f[i])
    f[i] = f0[i]
    fa = FAct[i] @ f0[i]
    Ia = fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]
    tmp = tm.mat3([(0., 0., 0.), (0., 0., 0.), (0., 0., 0.)])
    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            tmp[j, k] = f[i][j] * f[i][k]
    tmp = FAct[i] @ tmp
    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            ga[i][j * 3 + k] = 2 * tmp[j, k]

    R, S = ti.polar_decompose(FAct[i], dt=ti.f32)
    sign = f[i].transpose() @ S @ f[i]
    IIs = 0
    if ti.abs(sign[0]) < 1e-5:
        IIs = 0
    if sign[0] > 0:
        IIs = 1
    else:
        IIs = -1

    aniso_psi[i] = fiber_strength * LameMu[None] / 2 * (ti.sqrt(Ia) - IIs) * (ti.sqrt(Ia) - IIs)
    aniso_dpsi_vec[i] = (fiber_strength * LameMu[None] * (ti.sqrt(Ia) - IIs) / 2 / ti.sqrt(Ia)) * ga[i]
    # aniso_dpsi_vec[i] = dFactdF[i].transpose() @ aniso_dpsi_vec[i]
    tmp_vec = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for j in ti.static(range(9)):
        for k in ti.static(range(9)):
            tmp_vec[j] += dFactdFT[i, j, k] * aniso_dpsi_vec[i][k]
    for j in ti.static(range(9)):
        aniso_dpsi_vec[i][j] = tmp_vec[j]

    Psi[i] = aniso_psi[i] + iso_psi[i]
    if i == 1:
        print("aniso:", aniso_psi[i])
        print("iso:", iso_psi[i])
    # dpsi_vec[i] = iso_dpsi_vec[i] + dFactdF[i].transpose() @ aniso_dpsi_vec[i]
    # tmp_vec = ti.Vector([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # for j in ti.static(range(9)):
    #     for k in ti.static(range(9)):
    #         tmp_vec[j] += dFactdFT[i, j, k] * aniso_dpsi_vec[i][k]
    # for j in ti.static(range(9)):
    #     aniso_dpsi_vec[i][j] = tmp_vec[j]
    dpsi_vec[i] = iso_dpsi_vec[i] + aniso_dpsi_vec[i]

    for j in ti.static(range(3)):
        for k in ti.static(range(3)):
            P[i][j, k] = dpsi_vec[i][j * 3 + k]



@ti.kernel
def solveFEM(tet_set_id: ti.int32):
    for i in range(numTets):
        if tet_set[i] == tet_set_id:
            Ds[i][0, 0] = pos[tet[i][0]][0] - pos[tet[i][3]][0]
            Ds[i][1, 0] = pos[tet[i][0]][1] - pos[tet[i][3]][1]
            Ds[i][2, 0] = pos[tet[i][0]][2] - pos[tet[i][3]][2]
            Ds[i][0, 1] = pos[tet[i][1]][0] - pos[tet[i][3]][0]
            Ds[i][1, 1] = pos[tet[i][1]][1] - pos[tet[i][3]][1]
            Ds[i][2, 1] = pos[tet[i][1]][2] - pos[tet[i][3]][2]
            Ds[i][0, 2] = pos[tet[i][2]][0] - pos[tet[i][3]][0]
            Ds[i][1, 2] = pos[tet[i][2]][1] - pos[tet[i][3]][1]
            Ds[i][2, 2] = pos[tet[i][2]][2] - pos[tet[i][3]][2]
    #
    for i in range(numTets):
        if tet_set[i] == tet_set_id:
            getAct(i)

    for i in range(numTets):
        if tet_set[i] == tet_set_id:
            ComputePsiDeriv(i)

    for i in range(numTets):
        if tet_set[i] == tet_set_id:
            E[i] = restVol[i] * Psi[i]

    for i in range(numTets):
        if tet_set[i] == tet_set_id:
            dEdx012[i] = restVol[i] * P[i] @ invDmT[i]

            dEdx[i, 0][0] = dEdx012[i][0, 0]
            dEdx[i, 0][1] = dEdx012[i][1, 0]
            dEdx[i, 0][2] = dEdx012[i][2, 0]

            dEdx[i, 1][0] = dEdx012[i][0, 1]
            dEdx[i, 1][1] = dEdx012[i][1, 1]
            dEdx[i, 1][2] = dEdx012[i][2, 1]

            dEdx[i, 2][0] = dEdx012[i][0, 2]
            dEdx[i, 2][1] = dEdx012[i][1, 2]
            dEdx[i, 2][2] = dEdx012[i][2, 2]

            dEdx[i, 3] = - (dEdx[i, 0] + dEdx[i, 1] + dEdx[i, 2])

    for i in range(numTets):
        if tet_set[i] == tet_set_id:
            id = tm.ivec4(-1, -1, -1, -1)
            for j in ti.static(range(4)):
                id[j] = tet[i][j]
            w = invMass[id[0]] * (dEdx[i, 0].norm()) ** 2 + invMass[id[1]] * (dEdx[i, 1].norm()) ** 2 + invMass[
                id[2]] * (
                    dEdx[i, 2].norm()) ** 2 + invMass[id[3]] * (dEdx[i, 3].norm()) ** 2

            if w == 0.0:
                para_lambda[i] = 0.0
            else:
                para_lambda[i] = - E[i] / w

            # fb = FAct[i] @ (f0[i].cross(s0[i]))
            # fb = tm.vec3(0., 0., 1.)
            for j in ti.static(range(4)):
                pos[tet[i][j]] += invMass[tet[i][j]] * para_lambda[i] * dEdx[i, j]
                # deltax = invMass[tet[i][j]] * para_lambda[i] * dEdx[i, j]
                # len_fb = tm.length(fb)
                # deltax_ = deltax.dot(fb) * fb / len_fb / len_fb
                # pos[tet[i][j]] += deltax_



@ti.kernel
def solve_bound():
    for i in range(numParticles):
        if bou_tag[i] == 1:
            pos[i] = init_pos[i]



@ti.kernel
# 更新速度
def postSolve():
    for i in pos:
        vel[i] = (pos[i] - prevPos[i]) / dt


@ti.kernel
def updateAct():
    for i in range(numTets):
        # actval[i] = - 2.0 * frame[None] * frame[None] / 600. / 600. + 2 * frame[None] / 600.
        # actval[i] = 2.0 * frame[None] * frame[None] / 300. / 300. - 2 * frame[None] / 300.
        tmp = frame[None] * 50.0 / 600.0
        left = int(tmp)
        right = left + 1
        actval[i] = contraction[left] + (contraction[right] - contraction[left]) * (tmp - left) / (right - left)
        actval[i] *= 0.3

        # if i == 1:
        #     print(actval[i])


@ti.kernel
def updateFrame():
    frame[None] += 10
    if frame[None] >= 600:
        frame[None] -= 600

def substep():
    # updateFrame()
    updateAct()
    preSolve()
    solve()
    postSolve()


@ti.kernel
def init():
    frame[None] = 0
    # 修改位置
    for i in range(numParticles):
        pos[i] += tm.vec3(0.5, 0.5, 0)

    # 杨氏模量和泊松比
    YoungsModulus[None] = 1000
    PoissonsRatio[None] = 0.49

    # 初始化Ds
    for i in range(numTets):
        Ds[i][0, 0] = pos[tet[i][0]][0] - pos[tet[i][3]][0]
        Ds[i][1, 0] = pos[tet[i][0]][1] - pos[tet[i][3]][1]
        Ds[i][2, 0] = pos[tet[i][0]][2] - pos[tet[i][3]][2]
        Ds[i][0, 1] = pos[tet[i][1]][0] - pos[tet[i][3]][0]
        Ds[i][1, 1] = pos[tet[i][1]][1] - pos[tet[i][3]][1]
        Ds[i][2, 1] = pos[tet[i][1]][2] - pos[tet[i][3]][2]
        Ds[i][0, 2] = pos[tet[i][2]][0] - pos[tet[i][3]][0]
        Ds[i][1, 2] = pos[tet[i][2]][1] - pos[tet[i][3]][1]
        Ds[i][2, 2] = pos[tet[i][2]][2] - pos[tet[i][3]][2]
    # 初始化F
    for i in range(numTets):
        F[i] = Ds[i] @ invDm[i]

    for i in range(numTets):
        f[i] = F[i] @ f0[i]
        f[i] /= tm.length(f[i])
        s[i] = F[i] @ s0[i]
        s[i] /= tm.length(s[i])

    for i in range(numTets):
        C[i] = F[i].transpose() @ F[i]

    for i in range(numTets):
        I1ref[i] = C[i].trace()
        I4fref[i] = f[i].dot(C[i] @ f[i])

    # 初始化收缩率
    for i in range(numTets):
        actval[i] = 0

    for i in range(numParticles):
        init_pos[i] = pos[i]


# ---------------------------------------------------------------------------- #
#                                      gui                                     #
# ---------------------------------------------------------------------------- #
# init the window, canvas, scene and camerea
window = ti.ui.Window("pbd", (1024, 1024), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# initial camera position
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


# def main(timeVccc=None):



if __name__ == '__main__':
    # main()
    init()
    # maxy = -10000
    # for i in range(numParticles):
    #     if maxy < pos[i][1]:
    #         maxy = pos[i][1]
    # print(maxy)
    while window.running:
        updateFrame()
        # do the simulation in each step
        for _ in range(numSubsteps):
            substep()

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        # draw
        # scene.particles(pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(pos, indices=surf_show, color=(1, 0, 0), two_sided=False)

        # show the frame
        canvas.scene(scene)
        window.show()
