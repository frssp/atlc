#!/usr/bin/python

import numpy as np
import scipy.interpolate
import scipy.integrate
from scipy.integrate import cumtrapz, trapz
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd
import sys

# e = 1.60217662E-19 # elementary charge
# kb = 8.6173303e-5  # Boltzmann constant
sun_power = 100.   # mW/cm2

k = 1.38064852e-23     # m^2 kg s^-2 K^-1, Boltzmann constant
h = 6.62607004e-34     # m^2 kg s^-1    , planck constant
c = 2.99792458e8       # m s^-1         , speed of light
eV = 1.6021766208e-19  # joule        , eV to joule
q = 1.6021766208e-19   # C             , elemental charge

# http://rredc.nrel.gov/solar/spectra/am1.5/
ref_solar = pd.read_csv("ASTMG173.csv", header=1)  # nm vs W m^-2 nm^-1
# data range: 280nm to 4000nm, 0.31eV to 4.42857 eV
# WL (nm), W*m-2*nm-1
WL, solar_per_nm = ref_solar.iloc[:, 0], ref_solar.iloc[:, 2]
E = 1240.0 / WL  # eV
# jacobian transformation, W m^-2 eV^-1
solar_per_E = solar_per_nm * (eV/1e-9) * h * c / (eV*E)**2

Es = np.arange(0.32, 4.401, 0.002)

# linear interpolation to get an equally spaced spectrum
AM15 = np.interp(Es, E[::-1], solar_per_E[::-1])  # W m^-2 eV^-1
AM15flux = AM15 / (Es*eV)  # number of photon m^-2 eV^-1 s^-1


class tlc(object):
    ALPHA_FILE = "alpha.csv"

    def __init__(self, E_gap, T=300, thickness=2000, intensity=1.0):
        """
        E_gap: band gap in eV
        T: temperature in K
        thickness: thickness in nm
        intensity: light concentration, 1.0 = one Sun, 100 mW/cm^2
        """
        try:
            E_gap, T, thickness, intensity = float(E_gap), float(
                T), float(thickness), float(intensity)
        except:
            raise ValueError(
                "Invalid input for E_gap, T, thickness, or intensity")

        if T <= 0 or E_gap < 0.31:
            raise ValueError("T must be greater than 0 and " +
                             "E_gap cannot be less than 0.31")

        self.T = T
        self.E_gap = E_gap
        self.thickness = thickness
        self.intensity = intensity
        self.Es = Es  # np.arange(0.32, 4.401, 0.002)
        self.l_calc = False
        self._calc_absorptivity()
        # self.calculate()
        # self.plot_jv()
        # self.print_params()
        # self.WLs = np.arange(280, 4001, 1.0)
        # self.AM15nm = np.interp(self.WLs, WL, solar_per_nm)

    def __repr__(self):
        """
        params
        """
        s = "Trap limited conversion efficiency (TLC)\n"
        s += "T: {:.1f} K\n".format(self.T)
        s += "E_gap: {:.1f} eV\n".format(self.E_gap)
        s += "Thickness: {:.1f} nm".format(self.thickness)
        if self.l_calc:
            s += "\n===\n"
            s += "J_sc: {:.3f} mA/cm^2\n".format(self.j_sc)
            s += "J0_rad: {:.3g} mA/cm^2\n".format(self.j0_rad)
            s += "V_oc: {:.3f} V\n".format(self.v_oc)
            s += "V_max, J_max: {:.3f} V, {:.3f} mA/cm^2\n".format(
                self.v_max, self.j_max)
            s += "FF: {:.3f}%\n".format(self.ff*100)
            s += "Efficiency: {:.3f}%".format(self.efficiency*100)
        return s

    def calculate(self):
        Vs = np.arange(-0.1, self.E_gap, 0.001)
        self.j_sc = self.__cal_E_J_sc()
        self.j0_rad = self.__cal_J0_rad()
        self.jv = self.__cal_jv(Vs)
        self.v_oc = self.__cal_v_oc()
        self.v_max, self.j_max, self.efficiency = self.__calc_eff()
        self.ff = self.__calc_ff()
        self.l_calc = True

    def plot_jv(self):
        self.jv.mask(self.jv.J > 100).plot(x="V", y="J")
        plt.ylim((self.j_sc*-1.2, 0))
        plt.xlim((0, self.E_gap))
        plt.xlabel("Voltage (V)", fontsize=16)
        plt.ylabel("Current density (mA/$\mathregular{cm^2}$)",
                   fontsize=16)
        plt.title("Theoretical J-V for Eg = {:.3f} eV".format(self.E_gap))
        plt.show()

    def __cal_E_J_sc(self):
        fluxcumm = cumtrapz(
            self.absorptivity[::-1] * AM15flux[::-1], self.Es[::-1], initial=0)
        # fluxcumm = cumtrapz(AM15flux[::-1], self.Es[::-1], initial=0)
        # TODO: no E_gap
        fluxaboveE = fluxcumm[::-1] * -1 * self.intensity
        flux_absorbed = interp1d(self.Es, fluxaboveE)(self.E_gap)
        #
        J_sc = flux_absorbed * q * 0.1  # mA/cm^2  (0.1: from A/m2 to mA/cm2)
        return J_sc

    def __cal_J0_rad(self):
        '''
        Calculate and return J0, the dark saturation current
        J0 = q * (integrate(phi dE) from E to infinity)  / EQE_EL
        phi is the black body radiation at T (flux vs energy)
        '''
        phi = 2 * np.pi * (((self.Es*eV)**2) * eV / ((h**3) * (c**2)) / (
                           np.exp(self.Es*eV / (k*self.T)) - 1))

        # fluxcumm = cumtrapz(phi[::-1], self.Es[::-1], initial=0)
        fluxcumm = cumtrapz(
            self.absorptivity[::-1] * phi[::-1], self.Es[::-1], initial=0)
        # TODO: no E_gap
        fluxaboveE = fluxcumm[::-1] * -1
        flux_absorbed = interp1d(self.Es, fluxaboveE)(self.E_gap)
        #
        j0 = flux_absorbed * q * 0.1  # (0.1: from A/m2 to mA/cm2)
        return j0

    def __cal_jv(self, Vs):
        """
        simulate jv
        """
        j_sc, j0_rad = self.j_sc, self.j0_rad

        j = -1.0 * j_sc + j0_rad * (np.exp(q*Vs / (k*self.T)) - 1)
        jv = pd.DataFrame({"V": Vs, "J": j})
        return jv

    def __cal_v_oc(self):
        """
        """

        def f(v): return interp1d(self.jv.V, self.jv.J)(v)
        sol = root_scalar(f, bracket=[0, self.jv.V.max()])
        return sol.root

    def __find_max_point(self):
        """ 
        Calculate aren return the voltage that produces
        the maximum power
        """
        power = self.jv.J * self.jv.V
        def f(v): return interp1d(self.jv.V, power)(v)
        res = minimize_scalar(f, method='Bounded', bounds=[0, self.jv.V.max()])
        return res.x

    def __calc_eff(self):
        v_max = self.__find_max_point()
        power = self.jv.J * self.jv.V

        def eff(v): return interp1d(self.jv.V, power)(v) \
            / sun_power * self.intensity

        def j(v): return interp1d(self.jv.V, self.jv.J)(v)
        return v_max, -j(v_max), -eff(v_max)

    def __calc_ff(self):
        ff = self.v_max * self.j_max / self.v_oc / self.j_sc
        return ff

    def __read_alpha(self):
        alpha = pd.read_csv(self.ALPHA_FILE)
        # alpha.plot(x='E', y='alpha')
        self.alpha = alpha

    def _calc_absorptivity(self):
        self.__read_alpha()
        absorptivity = 1 - \
            np.exp(-2 * self.alpha.alpha * self.thickness * 1E7)  # nm -> cm
        self.absorptivity = np.interp(
            Es, self.alpha.E[::-1], absorptivity[::-1])  # W m^-2 eV^-1


    def plot_tauc(self):
        tauc = (self.alpha.alpha*self.alpha.E)**2
        plt.plot(self.alpha.E, tauc)
        plt.plot([self.E_gap, self.E_gap], [-1E10, 1E10], ls='--', label="Band gap")
        
        plt.xlabel("Energy (eV)", fontsize=16)
        plt.ylabel("$\mathregular{(ahv)^2}$ ($\mathregular{eV^2cm^{-2}}$)", fontsize=16)
        plt.title("Tauc plot")
        plt.legend()
        plt.xlim((self.E_gap-0.5, self.E_gap+0.5))
        plt.ylim((0, 10E9))
        # plt.yscale("log")
        plt.show() 

    def plot_alpha(self):
        self.alpha.plot(x='E', y='alpha', logy=True)
        plt.plot([self.E_gap, self.E_gap], [-1E10, 1E10], ls='--', label="Band gap")
        plt.ylim((10E0, 10E6))
        # plt.xlim((0, self.E_gap))
        plt.xlabel("Energy (eV)", fontsize=16)
        plt.ylabel("Absorption coefficient ($\mathregular{cm^{-1}}$)",
                   fontsize=16)
        plt.title("Absorption coefficient (taken from {})".format(self.ALPHA_FILE))
        plt.legend()
        plt.show() 

if __name__ == "__main__":
    tlc_CZTS = tlc(1.5, T=300)
    tlc_CZTS.calculate()
    print(tlc_CZTS)
    tlc_CZTS.plot_tauc()