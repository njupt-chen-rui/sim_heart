"""
    Body consists of vertexes and elements
"""
import taichi as ti
import numpy as np
import readVTK
import geometrytool as geo
import taichi.math as tm


@ti.data_oriented
class Body:
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

        # variables for visualization
        surfaces = geo.get_surface_from_tet(vertex=vertex, elements=elements)
        self.surfaces = ti.field(ti.i32, shape=(surfaces.shape[0] * surfaces.shape[1]))
        self.surfaces.from_numpy(surfaces.reshape(-1))
        self.Dm = ti.Matrix.field(3, 3, ti.f32, shape=(self.num_tet,))
        self.DmInv = ti.Matrix.field(3, 3, ti.f32, shape=(self.num_tet,))
        self.DmInvT = ti.Matrix.field(3, 3, ti.f32, shape=(self.num_tet,))
        self.init_DmInv()

        # 顶点fiber方向
        self.vert_fiber = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.vert_fiber.from_numpy(vert_fiber)
        # 四面体fiber方向
        self.tet_fiber = ti.Vector.field(3, float, shape=(self.num_tet,))
        # 从顶点采样到四面体
        self.sample_tet_fiber()

    @ti.kernel
    def init_DmInv(self):
        for i in range(self.num_tet):
            self.Dm[i][0, 0] = self.vertex[self.elements[i][0]][0] - self.vertex[self.elements[i][3]][0]
            self.Dm[i][1, 0] = self.vertex[self.elements[i][0]][1] - self.vertex[self.elements[i][3]][1]
            self.Dm[i][2, 0] = self.vertex[self.elements[i][0]][2] - self.vertex[self.elements[i][3]][2]
            self.Dm[i][0, 1] = self.vertex[self.elements[i][1]][0] - self.vertex[self.elements[i][3]][0]
            self.Dm[i][1, 1] = self.vertex[self.elements[i][1]][1] - self.vertex[self.elements[i][3]][1]
            self.Dm[i][2, 1] = self.vertex[self.elements[i][1]][2] - self.vertex[self.elements[i][3]][2]
            self.Dm[i][0, 2] = self.vertex[self.elements[i][2]][0] - self.vertex[self.elements[i][3]][0]
            self.Dm[i][1, 2] = self.vertex[self.elements[i][2]][1] - self.vertex[self.elements[i][3]][1]
            self.Dm[i][2, 2] = self.vertex[self.elements[i][2]][2] - self.vertex[self.elements[i][3]][2]

        for i in range(self.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()
            self.DmInvT[i] = self.DmInv[i].transpose()

    @ti.kernel
    def sample_tet_fiber(self):
        for i in range(self.num_tet):
            self.tet_fiber[i] = self.vert_fiber[self.elements[i][0]] + self.vert_fiber[self.elements[i][1]] + \
                                self.vert_fiber[self.elements[i][2]] + self.vert_fiber[self.elements[i][3]]
            self.tet_fiber[i] /= 4.0
            self.tet_fiber[i] /= tm.length(self.tet_fiber[i])

    def show(self):
        windowLength = 1024
        lengthScale = min(windowLength, 512)
        light_distance = lengthScale / 25.

        x_min = min(self.vertex[i][0] for i in range(self.vertex.shape[0]))
        x_max = max(self.vertex[i][0] for i in range(self.vertex.shape[0]))
        y_min = min(self.vertex[i][1] for i in range(self.vertex.shape[0]))
        y_max = max(self.vertex[i][1] for i in range(self.vertex.shape[0]))
        z_min = min(self.vertex[i][2] for i in range(self.vertex.shape[0]))
        z_max = max(self.vertex[i][2] for i in range(self.vertex.shape[0]))
        length = max(x_max - x_min, y_max - y_min, z_max - z_min)
        visualizeRatio = lengthScale / length / 10.
        center = np.array([(x_min + x_max) / 2., (y_min + y_max) / 2., (z_min + z_max) / 2.]) * visualizeRatio

        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # camera.position(0.5, 1.0, 1.95)
        camera.position(length * 0.4, length * 0.7, length * 2.)
        camera.lookat(center[0], center[1], center[2])
        camera.fov(55)
        while window.running:

            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # draw
            # scene.particles(pos, radius=0.02, color=(0, 1, 1))
            scene.mesh(self.vertex, indices=self.surfaces, color=(1, 0, 0), two_sided=False)

            # show the frame
            canvas.scene(scene)
            window.show()


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True)
    filename = "./data/heart1/heart_origin_bou_tag.vtk"
    points, mesh, bou_tag = readVTK.read_vtk(filename)
    body = Body(vertex=points, elements=mesh)

    body.show()
