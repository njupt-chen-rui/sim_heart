import readVTK
readVTK.read_vtk()

import taichi as ti
import numpy as np
import taichi.math as tm

ti.init(arch=ti.gpu)

