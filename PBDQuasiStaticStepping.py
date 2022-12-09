import taichi as ti
import numpy as np
from body import Body
import readVTK
from material import *
from mesh_data import heartMesh


@ti.data_oriented
class PBD_with_Continuous_Materials:
    def __init__(self, body: Body, material, num_sub_steps):
        self.body = body
        self.material = material
        self.vertex = body.vertex
        self.elements = body.elements
        self.num_tet = self.body.num_tet
        self.DmInv = body.DmInv
        self.Ds = ti.Matrix.field(3, 3, ti.f32, shape=(body.num_tet,))
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=(body.num_tet,))
        self.prevPos = ti.Vector.field(3, ti.f32, shape=(body.num_vertex,))
        self.vel = ti.Vector.field(3, ti.f32, shape=(body.num_vertex,))
        self.num_sub_steps = num_sub_steps
        self.dt = 1. / 60. / num_sub_steps

        # PBD所需的各位约束条件
        # 边长
        numEdges = len(heartMesh['tetEdgeIds']) // 2
        self.num_edges = numEdges
        edge_np = np.array(heartMesh['tetEdgeIds'], dtype=int)
        edge_np = edge_np.reshape((-1, 2))
        self.edge = ti.Vector.field(2, int, numEdges)
        self.edge.from_numpy(edge_np)

        # 能量
        self.Psi = ti.field(float, shape=(body.num_tet,))
        self.E = ti.field(float, shape=(body.num_tet,))
        self.P = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))

        # color graph
        self.max_edge_set = 23
        self.max_tet_set = 36
        edge_set_np = np.array(heartMesh['edge_set'], dtype=int)
        tet_set_np = np.array(heartMesh['tet_set'], dtype=int)
        self.edge_set = ti.field(ti.int32, numEdges)
        self.tet_set = ti.field(ti.int32, body.num_tet)
        self.edge_set.from_numpy(edge_set_np)
        self.tet_set.from_numpy(tet_set_np)

        # 初始状态
        self.restVol = ti.field(float, self.body.num_tet)
        self.restLen = ti.field(float, self.num_edges)
        self.invMass = ti.field(float, self.body.num_vertex)

        # 柔度
        self.edgeCompliance = 100.0
        self.volumeCompliance = 0.0
        self.init()

    # ---------------------------------------------------------------------------- #
    #                                      init                                     #
    # ---------------------------------------------------------------------------- #
    def init(self):
        self.init_pos()
        self.init_Ds()
        self.init_tet_volume()
        self.init_edge_len()
        self.init_invMass()

    @ti.kernel
    def init_pos(self):
        """通过平移调整物体位置"""
        for i in range(self.body.num_vertex):
            self.vertex[i] += tm.vec3(0.5, 0.5, 0)

    @ti.kernel
    def init_Ds(self):
        for i in range(self.body.num_tet):
            self.get_Ds(i)

    @ti.func
    def get_Ds(self, i):
        self.Ds[i][0, 0] = self.vertex[self.elements[i][0]][0] - self.vertex[self.elements[i][3]][0]
        self.Ds[i][1, 0] = self.vertex[self.elements[i][0]][1] - self.vertex[self.elements[i][3]][1]
        self.Ds[i][2, 0] = self.vertex[self.elements[i][0]][2] - self.vertex[self.elements[i][3]][2]
        self.Ds[i][0, 1] = self.vertex[self.elements[i][1]][0] - self.vertex[self.elements[i][3]][0]
        self.Ds[i][1, 1] = self.vertex[self.elements[i][1]][1] - self.vertex[self.elements[i][3]][1]
        self.Ds[i][2, 1] = self.vertex[self.elements[i][1]][2] - self.vertex[self.elements[i][3]][2]
        self.Ds[i][0, 2] = self.vertex[self.elements[i][2]][0] - self.vertex[self.elements[i][3]][0]
        self.Ds[i][1, 2] = self.vertex[self.elements[i][2]][1] - self.vertex[self.elements[i][3]][1]
        self.Ds[i][2, 2] = self.vertex[self.elements[i][2]][2] - self.vertex[self.elements[i][3]][2]

    @ti.kernel
    def init_tet_volume(self):
        for i in self.restVol:
            self.restVol[i] = self.get_tet_volume(i)

    @ti.func
    def get_tet_volume(self, i):
        id = tm.ivec4(-1, -1, -1, -1)
        for j in ti.static(range(4)):
            id[j] = self.elements[i][j]
        temp = (self.vertex[id[1]] - self.vertex[id[0]]).cross(self.vertex[id[2]] - self.vertex[id[0]])
        res = temp.dot(self.vertex[id[3]] - self.vertex[id[0]])
        res *= 1.0 / 6.0
        return res

    @ti.kernel
    def init_edge_len(self):
        for i in self.restLen:
            self.restLen[i] = self.get_edge_len(i)

    @ti.func
    def get_edge_len(self, i):
        res = (self.vertex[self.edge[i][0]] - self.vertex[self.edge[i][1]]).norm()
        return res

    @ti.kernel
    def init_invMass(self):
        for i in range(self.body.num_tet):
            pInvMass = 0.0
            if self.restVol[i] > 0.0:
                pInvMass = 1.0 / (self.restVol[i] / 4.0)
            for j in ti.static(range(4)):
                self.invMass[self.elements[i][j]] += pInvMass

    # ---------------------------------------------------------------------------- #
    #                            update during simulation                          #
    # ---------------------------------------------------------------------------- #
    def update(self):
        for _ in range(self.num_sub_steps):
            self.sub_step()

    def sub_step(self):
        self.pre_solve()
        self.solve()
        self.post_solve()

    @ti.kernel
    def pre_solve(self):
        """
        辛欧拉时间积分，计算外力的影响，以及和地面的碰撞
        """
        # g = tm.vec3(0, -9.8, 0)
        g = tm.vec3(0, 0, 0)
        for i in self.vertex:
            self.prevPos[i] = self.vertex[i]
            self.vel[i] += g * self.dt
            self.vertex[i] += self.vel[i] * self.dt
            if self.vertex[i].y < 0.:
                self.vertex[i] = self.prevPos[i]
                self.vertex[i].y = 0.

    def solve(self):
        """解算约束"""
        for i in range(1, self.max_tet_set + 1):
            self.solveFEM(i)

        for i in range(1, self.max_edge_set + 1):
            self.solveEdge(i)

        for i in range(1, self.max_tet_set + 1):
            self.solveVolume(i)

        # self.solve_bound()

    @ti.kernel
    def solveFEM(self, tet_set_id: ti.int32):
        for i in range(self.num_tet):
            if self.tet_set[i] == tet_set_id:
                self.get_Ds(i)
                self.F[i] = self.Ds[i] @ self.body.DmInv[i]
                self.Psi[i], self.P[i] = self.material.ComputePsiDeriv(self.F[i], self.body.tet_fiber[i])
                self.E[i] = self.restVol[i] * self.Psi[i]
                dEdx012 = self.restVol[i] * self.P[i] @ self.body.DmInvT[i]
                dEdx0 = tm.vec3([dEdx012[0, 0], dEdx012[1, 0], dEdx012[2, 0]])
                dEdx1 = tm.vec3([dEdx012[0, 1], dEdx012[1, 1], dEdx012[2, 1]])
                dEdx2 = tm.vec3([dEdx012[0, 2], dEdx012[1, 2], dEdx012[2, 2]])
                dEdx3 = - (dEdx0 + dEdx1 + dEdx2)
                dEdx = (dEdx0, dEdx1, dEdx2, dEdx3)
                id = tm.ivec4(-1, -1, -1, -1)
                for j in ti.static(range(4)):
                    id[j] = self.elements[i][j]
                w = self.invMass[id[0]] * (dEdx0.norm()) ** 2 + self.invMass[id[1]] * (dEdx1.norm()) ** 2 + \
                    self.invMass[id[2]] * (dEdx2.norm()) ** 2 + self.invMass[id[3]] * (dEdx3.norm()) ** 2
                para_lambda = 0.
                if w == 0.0:
                    para_lambda = 0.0
                else:
                    para_lambda = - self.E[i] / w

                self.vertex[self.elements[i][0]] += self.invMass[self.elements[i][0]] * para_lambda * dEdx0
                self.vertex[self.elements[i][1]] += self.invMass[self.elements[i][1]] * para_lambda * dEdx1
                self.vertex[self.elements[i][2]] += self.invMass[self.elements[i][2]] * para_lambda * dEdx2
                self.vertex[self.elements[i][3]] += self.invMass[self.elements[i][3]] * para_lambda * dEdx3

    @ti.kernel
    def solveEdge(self, edge_set_id: ti.int32):
        alpha = self.edgeCompliance / self.dt / self.dt
        grads = tm.vec3(0, 0, 0)
        for i in range(self.num_edges):
            if self.edge_set[i] == edge_set_id:
                id0 = self.edge[i][0]
                id1 = self.edge[i][1]

                grads = self.vertex[id0] - self.vertex[id1]
                Len = grads.norm()
                grads = grads / Len
                C = Len - self.restLen[i]
                w = self.invMass[id0] + self.invMass[id1]
                s = -C / (w + alpha)

                self.vertex[id0] += grads * s * self.invMass[id0]
                self.vertex[id1] += grads * (-s * self.invMass[id1])

    @ti.kernel
    def solveVolume(self, tet_set_id: ti.int32):
        alpha = self.volumeCompliance / self.dt / self.dt
        grads = [tm.vec3(0, 0, 0), tm.vec3(0, 0, 0), tm.vec3(0, 0, 0), tm.vec3(0, 0, 0)]

        for i in range(self.body.num_tet):
            if self.tet_set[i] == tet_set_id:
                id = tm.ivec4(-1, -1, -1, -1)
                for j in ti.static(range(4)):
                    id[j] = self.elements[i][j]
                grads[0] = (self.vertex[id[3]] - self.vertex[id[1]]).cross(self.vertex[id[2]] - self.vertex[id[1]])
                grads[1] = (self.vertex[id[2]] - self.vertex[id[0]]).cross(self.vertex[id[3]] - self.vertex[id[0]])
                grads[2] = (self.vertex[id[3]] - self.vertex[id[0]]).cross(self.vertex[id[1]] - self.vertex[id[0]])
                grads[3] = (self.vertex[id[1]] - self.vertex[id[0]]).cross(self.vertex[id[2]] - self.vertex[id[0]])

                w = 0.0
                for j in ti.static(range(4)):
                    w += self.invMass[id[j]] * (grads[j].norm()) ** 2

                vol = self.get_tet_volume(i)
                C = (vol - self.restVol[i]) * 6.0
                s = -C / (w + alpha)

                for j in ti.static(range(4)):
                    self.vertex[self.elements[i][j]] += grads[j] * s * self.invMass[id[j]]

    @ti.kernel
    def post_solve(self):
        """更新速度"""
        for i in self.vertex:
            self.vel[i] = (self.vertex[i] - self.prevPos[i]) / self.dt

    def show(self):
        window = ti.ui.Window("PBD_with_Continuous_Materials", (1024, 1024), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # initial camera position
        camera.position(0.5, 1.0, 1.95)
        camera.lookat(0.5, 0.3, 0.5)
        camera.fov(55)
        while window.running:

            # do the simulation in each step
            self.update()

            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
            scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
            scene.ambient_light((0.5, 0.5, 0.5))

            # draw
            # scene.particles(pos, radius=0.02, color=(0, 1, 1))
            scene.mesh(self.vertex, indices=self.body.surfaces, color=(1, 0, 0), two_sided=False)

            # show the frame
            canvas.scene(scene)
            window.show()


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
    body = Body(pos_np, tet_np)
    Youngs_Modulus = 1000.
    Poisson_Ratio = 0.49
    material = Stable_Neo_Hookean(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio)
    sys = PBD_with_Continuous_Materials(body, material, 10)
    sys.show()
    # body.show()
