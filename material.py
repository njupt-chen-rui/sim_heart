import numpy as np
import taichi as ti
import taichi.math as tm


@ti.data_oriented
class NeoHookean:
    """
    elastic energy density:
    ψ = C1 * (I1 - 3 - 2 * ln(J)) + D1 * (J - 1)**2,
    σ = J^(-1) * ∂ψ/∂F * F^T = 2*C1*J^(-1)*(B - I) + 2*D1*(J-1)*I
    https://en.wikipedia.org/wiki/Neo-Hookean_solid
    """
    def __init__(self, Youngs_modulus, Poisson_ratio):
        self.Youngs_modulus = Youngs_modulus
        self.Poisson_ratio = Poisson_ratio
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
        self.C1 = self.LameMu / 2.
        self.D1 = self.LameLa / 2.

    @ti.kernel
    def constitutive_small_deform(self, deformationGradient: ti.template(),
                                 cauchy_stress: ti.template()):
        # C1, D1 = ti.static(self.C1, self.D1)
        mu, la = ti.static(self.LameMu, self.LameLa)
        identity3 = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        for i in ti.grouped(deformationGradient):
            F = deformationGradient[i]
            J = F.determinant()
            B = F @ F.transpose()  # left Cauchy-Green Strain tensor
            cauchy_stress[i] = mu / J * (B - identity3) + la * (J - 1.) * identity3

    @ti.kernel
    def constitutive_large_deform(self, deformationGradient: ti.template(),
                                  cauchy_stress: ti.template()):
        # C1, D1 = ti.static(self.C1, self.D1)
        mu, la = ti.static(self.LameMu, self.LameLa)
        identity3 = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        for i in ti.grouped(deformationGradient):
            F = deformationGradient[i]
            J = F.determinant()
            B = F @ F.transpose()  # left Cauchy-Green Strain tensor
            cauchy_stress[i] = mu / J * (B - identity3) + la * (J - 1.) * identity3

    @ti.func
    def elastic_energy_density(self, deformationGradient):
        F = deformationGradient
        J = F.determinant()
        B = F @ F.transpose()
        return self.C1 * (B.trace() - 3. - 2. * ti.log(J)) + self.D1 * (J - 1.)**2


@ti.data_oriented
class Stable_Neo_Hookean:
    """
    elastic energy density:
    I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
    ψ = μ/2 * (I2-3) - μ(I3-1) + λ/2 * (I3-1)^2
    """
    def __init__(self, Youngs_modulus, Poisson_ratio):
        self.Youngs_modulus = Youngs_modulus
        self.Poisson_ratio = Poisson_ratio
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))

    @ti.func
    def ComputePsiDeriv(self, deformation_gradient: ti.template(), fiber_direction: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi and the first Piola-Kirchhoff tensor P
        """
        mu, la = ti.static(self.LameMu, self.LameLa)

        F = deformation_gradient
        J = F.determinant()

        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]

        # 定义不变量: I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
        I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
        I2 = sigma[0, 0] * sigma[0, 0] + sigma[1, 1] * sigma[1, 1] + sigma[2, 2] * sigma[2, 2]
        I3 = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]

        # 定义不变量对于F的导数
        R = U @ V.transpose()
        col0 = tm.vec3(F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1],
                        F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2],
                        F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
        col1 = tm.vec3(F[2, 1] * F[0, 2] - F[2, 2] * F[0, 1],
                        F[2, 2] * F[0, 0] - F[2, 0] * F[0, 2],
                        F[2, 0] * F[0, 1] - F[2, 1] * F[0, 0])
        col2 = tm.vec3(F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1],
                        F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2],
                        F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
        dI1dF = R
        dI2dF = 2 * F
        dI3dF = tm.mat3([col0, col1, col2])

        # 定义能量密度
        # ψ = μ / 2 * (I2 - 3) - μ(I3 - 1) + λ / 2 * (I3 - 1) ^ 2
        Psi = mu / 2. * (I2 - 3.) - mu * (I3 - 1.) + la / 2. * (I3 - 1.) * (I3 - 1.)

        # 定义1st Piola-Kirchhoff tensor
        # P = μ / 2 * dI2dF - μ * dI3dF + λ * (I3 - 1) * dI3dF
        P = mu / 2. * dI2dF - mu * dI3dF + la * (I3 - 1.) * dI3dF

        return Psi, P


@ti.data_oriented
class Stable_Neo_Hookean_with_active:
    """
    elastic energy density:
    I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
    ψ = μ/2 * (I2-3) - μ(I3-1) + λ/2 * (I3-1)^2
    """
    def __init__(self, Youngs_modulus, Poisson_ratio, active_tension):
        self.Youngs_modulus = Youngs_modulus
        self.Poisson_ratio = Poisson_ratio
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
        self.Ta = active_tension

    @ti.func
    def ComputePsiDeriv(self, deformation_gradient: ti.template(), fiber_direction: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi and the first Piola-Kirchhoff tensor P
        """
        mu, la = ti.static(self.LameMu, self.LameLa)

        F = deformation_gradient
        f0 = fiber_direction
        # J = F.determinant()

        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]

        # 定义不变量: I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
        I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
        I2 = sigma[0, 0] * sigma[0, 0] + sigma[1, 1] * sigma[1, 1] + sigma[2, 2] * sigma[2, 2]
        I3 = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]

        # 定义不变量对于F的导数
        R = U @ V.transpose()
        col0 = tm.vec3(F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1],
                        F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2],
                        F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
        col1 = tm.vec3(F[2, 1] * F[0, 2] - F[2, 2] * F[0, 1],
                        F[2, 2] * F[0, 0] - F[2, 0] * F[0, 2],
                        F[2, 0] * F[0, 1] - F[2, 1] * F[0, 0])
        col2 = tm.vec3(F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1],
                        F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2],
                        F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
        dI1dF = R
        dI2dF = 2 * F
        dI3dF = tm.mat3([col0, col1, col2])

        # 定义能量密度
        # ψ = μ / 2 * (I2 - 3) - μ(I3 - 1) + λ / 2 * (I3 - 1) ^ 2
        Psi = mu / 2. * (I2 - 3.) - mu * (I3 - 1.) + la / 2. * (I3 - 1.) * (I3 - 1.)

        # 定义1st Piola-Kirchhoff tensor
        # P_pass = μ / 2 * dI2dF - μ * dI3dF + λ * (I3 - 1) * dI3dF
        P_pass = mu / 2. * dI2dF - mu * dI3dF + la * (I3 - 1.) * dI3dF
        # P_act = Ta * (F@f0)@(f0^T) / sqrt(I4f)
        f = (F @ f0)
        I4f = f[0] * f[0] + f[1] * f[1] + f[2] * f[2]
        P_act = self.Ta * (F @ f0) @ (f0.transpose()) / tm.sqrt(I4f)
        P = P_pass + P_act

        return Psi, P


@ti.kernel
def debug(material: ti.template()):
    F = tm.mat3([1, 0, 0,
                 0, 1, 0,
                 0, 0, 1])
    f0 = tm.vec3([1, 0, 0])
    Psi, P = material.ComputePsiDeriv(F, f0)
    print(Psi, P)


if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    Youngs_Modulus = 1000.
    Poisson_Ratio = 0.49
    # material = Stable_Neo_Hookean(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio)
    material = Stable_Neo_Hookean_with_active(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio,
                                              active_tension=60)
    debug(material)


