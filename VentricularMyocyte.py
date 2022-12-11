"""
    Single cell electrophysiological model
"""
import taichi as ti
import numpy as np
import taichi.math as tm
import time




@ti.data_oriented
class TP:
    def __init__(self, cell_type, num_tet) -> None:
        self.num_cell = num_tet
        self.cell_type = cell_type
        self.knak = 2.724
        self.KmNa = 40.0
        self.KmK = 1.0
        self.knaca = 1000
        self.KmNai = 87.5
        self.KmCa = 1.38
        self.ksat = 0.1
        self.n = 0.35

        self.Ko = 5.4
        self.Cao = 2.0
        self.Nao = 140.0

        self.Bufc = 0.2
        self.Kbufc = 0.001
        self.Bufsr = 10.
        self.Kbufsr = 0.3
        self.Bufss = 0.4
        self.Kbufss = 0.00025

        self.Vmaxup = 0.006375
        self.Kup = 0.00025
        self.Vrel = 0.102
        self.k1_ = 0.15
        self.k2_ = 0.045
        self.k3 = 0.060
        self.k4 = 0.005
        self.EC = 1.5
        self.maxsr = 2.5
        self.minsr = 1.
        self.Vleak = 0.00036
        self.Vxfer = 0.0038

        self.pKNa = 0.03

        self.CAPACITANCE = 0.185
        self.F = 96485.3415
        self.R = 8314.472
        self.T = 310.0
        self.RTONF = (self.R * self.T) / self.F

        self.Gkr = 0.153
        if self.cell_type == "MCELL":
            self.Gks = 0.098
        else:
            self.Gks = 0.392
        self.GK1 = 5.405
        if self.cell_type == "ENDO":
            self.Gto = 0.073
        else:
            self.Gto = 0.294
        self.GNa = 14.838
        self.GbNa = 0.00029
        self.GCaL = 0.00003980
        self.GbCa = 0.000592
        self.GpCa = 0.1238
        self.KpCa = 0.0005
        self.GpK = 0.0146

        self.Vc = 0.016404
        self.Vsr = 0.001094
        self.Vss = 0.00005468

        # self.period

        self.V_init = -86.2
        self.Cai_init = 0.00007
        self.CaSR_init = 1.3
        self.CaSS_init = 0.00007
        self.Nai_init = 7.67
        self.Ki_init = 138.3

        # 状态变量
        # voltage + backup
        self.Var_Volt = ti.field(float, shape=(self.num_cell,))
        self.Var_Volt2 = ti.field(float, shape=(self.num_cell,))
        self.Var_Cai = ti.field(float, shape=(self.num_cell,))
        self.Var_CaSR = ti.field(float, shape=(self.num_cell,))
        self.Var_CaSS = ti.field(float, shape=(self.num_cell,))
        self.Var_Nai = ti.field(float, shape=(self.num_cell,))
        # states of voltage and time dependent gates
        # INa
        self.Var_Ki = ti.field(float, shape=(self.num_cell,))
        self.Var_M = ti.field(float, shape=(self.num_cell,))
        self.Var_H = ti.field(float, shape=(self.num_cell,))
        self.Var_J = ti.field(float, shape=(self.num_cell,))
        # IKr
        # IKr1
        self.Var_Xr1 = ti.field(float, shape=(self.num_cell,))
        # IKr2
        self.Var_Xr2 = ti.field(float, shape=(self.num_cell,))
        # IKs
        self.Var_Xs = ti.field(float, shape=(self.num_cell,))
        # Ito1
        self.Var_R = ti.field(float, shape=(self.num_cell,))
        self.Var_S = ti.field(float, shape=(self.num_cell,))
        # ICa
        self.Var_D = ti.field(float, shape=(self.num_cell,))
        self.Var_F = ti.field(float, shape=(self.num_cell,))
        self.Var_F2 = ti.field(float, shape=(self.num_cell,))
        self.Var_FCass = ti.field(float, shape=(self.num_cell,))
        # Irel
        self.Var_RR = ti.field(float, shape=(self.num_cell,))
        self.Var_OO = ti.field(float, shape=(self.num_cell,))
        # total current
        self.Var_Itot = ti.field(float, shape=(self.num_cell,))
        self.variables_init()

    @ti.kernel
    def variables_init(self):
        for i in range(self.num_cell):
            self.Var_Volt[i] = self.V_init
            self.Var_Volt2[i] = self.V_init
            self.Var_Cai[i] = self.Cai_init
            self.Var_CaSR[i] = self.CaSR_init
            self.Var_CaSS[i] = self.CaSS_init
            self.Var_Nai[i] = self.Nai_init
            self.Var_Ki[i] = self.Ki_init
            self.Var_M[i] = 0.
            self.Var_H[i] = 0.75
            self.Var_J[i] = 0.75
            self.Var_Xr1[i] = 0.
            self.Var_Xr2[i] = 1.
            self.Var_Xs[i] = 0.
            self.Var_R[i] = 0.
            self.Var_S[i] = 1.
            self.Var_D[i] = 0.
            self.Var_F[i] = 1.
            self.Var_F2[i] = 1.
            self.Var_FCass[i] = 1.
            self.Var_RR[i] = 1.
            self.Var_OO[i] = 0.

    @ti.func
    def Step(self, i):
        # 输入
        Istim = 0.
        HT = 0.01

        inverseVcF2 = 1 / (2 * self.Vc * self.F)
        inverseVcF = 1. / (self.Vc * self.F)
        inversevssF2 = 1 / (2 * self.Vss * self.F)

        # 宏定义状态变量的别名
        sm, sh, sj = ti.static(self.Var_M, self.Var_H, self.Var_J)
        sxr1, sxr2, sxs = ti.static(self.Var_Xr1, self.Var_Xr2, self.Var_Xs)
        ss, sr, sd, sf, sf2 = ti.static(self.Var_S, self.Var_R, self.Var_D, self.Var_F, self.Var_F2)
        sfcass, sRR, sOO, = ti.static(self.Var_FCass, self.Var_RR, self.Var_OO)
        svolt, svolt2 = ti.static(self.Var_Volt, self.Var_Volt2)
        Cai, CaSR, CaSS, Nai, Ki = ti.static(self.Var_Cai, self.Var_CaSR, self.Var_CaSS, self.Var_Nai, self.Var_Ki)
        sItot = ti.static(self.Var_Itot)

        # Needed to compute currents
        Ek = self.RTONF * (tm.log((self.Ko / Ki[i])))
        Ena = self.RTONF * (tm.log((self.Nao / Nai[i])))
        Eks = self.RTONF * (tm.log((self.Ko + self.pKNa * self.Nao) / (Ki[i] + self.pKNa * Nai[i])))
        Eca = 0.5 * self.RTONF * (tm.log((self.Cao / Cai[i])))
        Ak1 = 0.1 / (1. + tm.exp(0.06 * (svolt[i] - Ek - 200)))
        Bk1 = (3. * tm.exp(0.0002 * (svolt[i] - Ek + 100)) +
               tm.exp(0.1 * (svolt[i] - Ek - 10))) / (1. + tm.exp(-0.5 * (svolt[i] - Ek)))
        rec_iK1 = Ak1 / (Ak1 + Bk1)
        rec_iNaK = (1. / (1. + 0.1245 * tm.exp(-0.1 * svolt[i] * self.F / (self.R * self.T)) + 0.0353 * tm.exp(
            -svolt[i] * self.F / (self.R * self.T))))
        rec_ipK = 1. / (1. + tm.exp((25 - svolt[i]) / 5.98))

        # Compute currents
        INa = self.GNa * sm[i] * sm[i] * sm[i] * sh[i] * sj[i] * (svolt[i] - Ena)
        ICaL = self.GCaL * sd[i] * sf[i] * sf2[i] * sfcass[i] * 4 * (svolt[i] - 15) * (
                    self.F * self.F / (self.R * self.T)) * (
                           0.25 * tm.exp(2 * (svolt[i] - 15) * self.F / (self.R * self.T)) * CaSS[i] - self.Cao) / (
                           tm.exp(2 * (svolt[i] - 15) * self.F / (self.R * self.T)) - 1.)
        Ito = self.Gto * sr[i] * ss[i] * (svolt[i] - Ek)
        IKr = self.Gkr * tm.sqrt(self.Ko / 5.4) * sxr1[i] * sxr2[i] * (svolt[i] - Ek)
        IKs = self.Gks * sxs[i] * sxs[i] * (svolt[i] - Eks)
        IK1 = self.GK1 * tm.sqrt(self.Ko / 5.4) * rec_iK1 * (svolt[i] - Ek)
        INaCa = self.knaca * (1. / (self.KmNai * self.KmNai * self.KmNai + self.Nao * self.Nao * self.Nao)) * (
                    1. / (self.KmCa + self.Cao)) * (
                            1. / (1 + self.ksat * tm.exp((self.n - 1) * svolt[i] * self.F / (self.R * self.T)))) * (
                            tm.exp(self.n * svolt[i] * self.F / (
                                        self.R * self.T)) * Nai[i] * Nai[i] * Nai[i] * self.Cao - tm.exp(
                        (self.n - 1) * svolt[i] * self.F / (
                                    self.R * self.T)) * self.Nao * self.Nao * self.Nao * Cai[i] * 2.5)
        INaK = self.knak * (self.Ko / (self.Ko + self.KmK)) * (Nai[i] / (Nai[i] + self.KmNa)) * rec_iNaK
        IpCa = self.GpCa * Cai[i] / (self.KpCa + Cai[i])
        IpK = self.GpK * rec_ipK * (svolt[i] - Ek)
        IbNa = self.GbNa * (svolt[i] - Ena)
        IbCa = self.GbCa * (svolt[i] - Eca)

        # Determine total current
        sItot[i] = IKr + IKs + IK1 + Ito + INa + IbNa + ICaL + IbCa + INaK + INaCa + IpCa + IpK + Istim

        # update concentrations
        kCaSR = self.maxsr - ((self.maxsr - self.minsr) / (1 + (self.EC / CaSR[i]) * (self.EC / CaSR[i])))
        k1 = self.k1_ / kCaSR
        k2 = self.k2_ * kCaSR
        dRR = self.k4 * (1 - sRR[i]) - k2 * CaSS[i] * sRR[i]
        sRR[i] += HT * dRR
        sOO[i] = k1 * CaSS[i] * CaSS[i] * sRR[i] / (self.k3 + k1 * CaSS[i] * CaSS[i])

        Irel = self.Vrel * sOO[i] * (CaSR[i] - CaSS[i])
        Ileak = self.Vleak * (CaSR[i] - Cai[i])
        Iup = self.Vmaxup / (1. + ((self.Kup * self.Kup) / (Cai[i] * Cai[i])))
        Ixfer = self.Vxfer * (CaSS[i] - Cai[i])

        CaCSQN = self.Bufsr * CaSR[i] / (CaSR[i] + self.Kbufsr)
        dCaSR = HT * (Iup - Irel - Ileak)
        bjsr = self.Bufsr - CaCSQN - dCaSR - CaSR[i] + self.Kbufsr
        cjsr = self.Kbufsr * (CaCSQN + dCaSR + CaSR[i])
        CaSR[i] = (tm.sqrt(bjsr * bjsr + 4 * cjsr) - bjsr) / 2.

        CaSSBuf = self.Bufss * CaSS[i] / (CaSS[i] + self.Kbufss)
        dCaSS = HT * (-Ixfer * (self.Vc / self.Vss) + Irel * (self.Vsr / self.Vss) + (
                    -ICaL * inversevssF2 * self.CAPACITANCE))
        bcss = self.Bufss - CaSSBuf - dCaSS - CaSS[i] + self.Kbufss
        ccss = self.Kbufss * (CaSSBuf + dCaSS + CaSS[i])
        CaSS[i] = (tm.sqrt(bcss * bcss + 4 * ccss) - bcss) / 2.

        CaBuf = self.Bufc * Cai[i] / (Cai[i] + self.Kbufc)
        dCai = HT * ((-(IbCa + IpCa - 2 * INaCa) * inverseVcF2 * self.CAPACITANCE) - (Iup - Ileak) * (
                    self.Vsr / self.Vc) + Ixfer)
        bc = self.Bufc - CaBuf - dCai - Cai[i] + self.Kbufc
        cc = self.Kbufc * (CaBuf + dCai + Cai[i])
        Cai[i] = (tm.sqrt(bc * bc + 4 * cc) - bc) / 2.

        dNai = -(INa + IbNa + 3 * INaK + 3 * INaCa) * inverseVcF * self.CAPACITANCE
        Nai[i] += HT * dNai

        dKi = -(Istim + IK1 + Ito + IKr + IKs - 2 * INaK + IpK) * inverseVcF * self.CAPACITANCE
        Ki[i] += HT * dKi

        # compute steady state values and time constants
        AM = 1. / (1. + tm.exp((-60. - svolt[i]) / 5.))
        BM = 0.1 / (1. + tm.exp((svolt[i] + 35.) / 5.)) + 0.10 / (1. + tm.exp((svolt[i] - 50.) / 200.))
        TAU_M = AM * BM
        M_INF = 1. / ((1. + tm.exp((-56.86 - svolt[i]) / 9.03)) * (1. + tm.exp((-56.86 - svolt[i]) / 9.03)))
        TAU_H = 0.
        if svolt[i] >= -40.:
            AH_1 = 0.
            BH_1 = (0.77 / (0.13 * (1. + tm.exp(-(svolt[i] + 10.66) / 11.1))))
            TAU_H = 1.0 / (AH_1 + BH_1)
        else:
            AH_2 = (0.057 * tm.exp(-(svolt[i] + 80.) / 6.8))
            BH_2 = (2.7 * tm.exp(0.079 * svolt[i]) + 310000 * tm.exp(0.3485 * svolt[i]))
            TAU_H = 1.0 / (AH_2 + BH_2)

        H_INF = 1. / ((1. + tm.exp((svolt[i] + 71.55) / 7.43)) * (1. + tm.exp((svolt[i] + 71.55) / 7.43)))
        TAU_J = 0.
        if svolt[i] >= -40.:
            AJ_1 = 0.
            BJ_1 = (0.6 * tm.exp(0.057 * svolt[i]) / (1. + tm.exp(-0.1 * (svolt[i] + 32.))))
            TAU_J = 1.0 / (AJ_1 + BJ_1)
        else:
            AJ_2 = (((-2.5428e4) * tm.exp(0.2444 * svolt[i]) - 6.948e-6 * tm.exp(-0.04391 * svolt[i])) * (
                        svolt[i] + 37.78) / (1. + tm.exp(0.311 * (svolt[i] + 79.23))))
            BJ_2 = (0.02424 * tm.exp(-0.01052 * svolt[i]) / (1. + tm.exp(-0.1378 * (svolt[i] + 40.14))))
            TAU_J = 1.0 / (AJ_2 + BJ_2)
        J_INF = H_INF

        Xr1_INF = 1. / (1. + tm.exp((-26. - svolt[i]) / 7.))
        axr1 = 450. / (1. + tm.exp((-45. - svolt[i]) / 10.))
        bxr1 = 6. / (1. + tm.exp((svolt[i] - (-30.)) / 11.5))
        TAU_Xr1 = axr1 * bxr1
        Xr2_INF = 1. / (1. + tm.exp((svolt[i] - (-88.)) / 24.))
        axr2 = 3. / (1. + tm.exp((-60. - svolt[i]) / 20.))
        bxr2 = 1.12 / (1. + tm.exp((svolt[i] - 60.) / 20.))
        TAU_Xr2 = axr2 * bxr2

        Xs_INF = 1. / (1. + tm.exp((-5. - svolt[i]) / 14.))
        Axs = (1400. / (tm.sqrt(1. + tm.exp((5. - svolt[i]) / 6))))
        Bxs = (1. / (1. + tm.exp((svolt[i] - 35.) / 15.)))
        TAU_Xs = Axs * Bxs + 80

        R_INF = 0.
        S_INF = 0.
        TAU_R = 0.
        TAU_S = 0.
        if self.cell_type == "ENDO":
            R_INF = 1. / (1. + tm.exp((20 - svolt[i]) / 6.))
            S_INF = 1. / (1. + tm.exp((svolt[i] + 28) / 5.))
            TAU_R = 9.5 * tm.exp(-(svolt[i] + 40.) * (svolt[i] + 40.) / 1800.) + 0.8
            TAU_S = 1000. * tm.exp(-(svolt[i] + 67) * (svolt[i] + 67) / 1000.) + 8.
        else:
            R_INF = 1. / (1. + tm.exp((20 - svolt[i]) / 6.))
            S_INF = 1. / (1. + tm.exp((svolt[i] + 20) / 5.))
            TAU_R = 9.5 * tm.exp(-(svolt[i] + 40.) * (svolt[i] + 40.) / 1800.) + 0.8
            TAU_S = 85. * tm.exp(-(svolt[i] + 45.) * (svolt[i] + 45.) / 320.) + 5. / (
                        1. + tm.exp((svolt[i] - 20.) / 5.)) + 3.

        D_INF = 1. / (1. + tm.exp((-8 - svolt[i]) / 7.5))
        Ad = 1.4 / (1. + tm.exp((-35 - svolt[i]) / 13)) + 0.25
        Bd = 1.4 / (1. + tm.exp((svolt[i] + 5) / 5))
        Cd = 1. / (1. + tm.exp((50 - svolt[i]) / 20))
        TAU_D = Ad * Bd + Cd
        F_INF = 1. / (1. + tm.exp((svolt[i] + 20) / 7))
        Af = 1102.5 * tm.exp(-(svolt[i] + 27) * (svolt[i] + 27) / 225)
        Bf = 200. / (1 + tm.exp((13 - svolt[i]) / 10.))
        Cf = (180. / (1 + tm.exp((svolt[i] + 30) / 10))) + 20
        TAU_F = Af + Bf + Cf
        F2_INF = 0.67 / (1. + tm.exp((svolt[i] + 35) / 7)) + 0.33
        Af2 = 600 * tm.exp(-(svolt[i] + 25) * (svolt[i] + 25) / 170)
        Bf2 = 31 / (1. + tm.exp((25 - svolt[i]) / 10))
        Cf2 = 16 / (1. + tm.exp((svolt[i] + 30) / 10))
        TAU_F2 = Af2 + Bf2 + Cf2
        FCaSS_INF = 0.6 / (1 + (CaSS[i] / 0.05) * (CaSS[i] / 0.05)) + 0.4
        TAU_FCaSS = 80. / (1 + (CaSS[i] / 0.05) * (CaSS[i] / 0.05)) + 2.

        # Update gates
        sm[i] = M_INF - (M_INF - sm[i]) * tm.exp(-HT / TAU_M)
        sh[i] = H_INF - (H_INF - sh[i]) * tm.exp(-HT / TAU_H)
        sj[i] = J_INF - (J_INF - sj[i]) * tm.exp(-HT / TAU_J)
        sxr1[i] = Xr1_INF - (Xr1_INF - sxr1[i]) * tm.exp(-HT / TAU_Xr1)
        sxr2[i] = Xr2_INF - (Xr2_INF - sxr2[i]) * tm.exp(-HT / TAU_Xr2)
        sxs[i] = Xs_INF - (Xs_INF - sxs[i]) * tm.exp(-HT / TAU_Xs)
        sr[i] = R_INF - (R_INF - sr[i]) * tm.exp(-HT / TAU_R)
        ss[i] = S_INF - (S_INF - ss[i]) * tm.exp(-HT / TAU_S)
        sdv = D_INF - (D_INF - sd[i]) * tm.exp(-HT / TAU_D)
        sf[i] = F_INF - (F_INF - sf[i]) * tm.exp(-HT / TAU_F)
        sf2[i] = F2_INF - (F2_INF - sf2[i]) * tm.exp(-HT / TAU_F2)
        sfcass[i] = FCaSS_INF - (FCaSS_INF - sfcass[i]) * tm.exp(-HT / TAU_FCaSS)

        # update voltage
        svolt[i] = svolt[i] + HT * (-sItot[i])

    @ti.kernel
    def sim(self):
        for i in range(self.num_cell):
            self.Step(i)


if __name__ == "__main__":
    ti.init(arch=ti.cuda, dynamic_index=True)
    cell_type = "MCELL"
    tp = TP(cell_type=cell_type, num_tet=800*800)
    start = time.time()
    for _ in range(800*50*5):
        tp.sim()
    end = time.time()
    print(end-start)

