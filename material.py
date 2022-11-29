import numpy as np
import taichi as ti


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
