import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate

from NuRadioReco.utilities import units
from NuRadioMC.utilities import fluxes

from multimessenger_tools.sensitivity import SensitivityCalculator, calculate_fluence_limit


def get_ic_aeff_boat():
    energy, aeff = np.genfromtxt("data/IC-GFU-GRB221009-aeff.csv", unpack="True", delimiter=",")
    energy *= units.GeV
    aeff *= units.m2
    a_eff = interpolate.interp1d(energy, aeff)
    return energy, aeff, a_eff


def plot_ic_aeff(ax):
    energy, aeff, _ = get_ic_aeff_boat()
    ax.plot(energy / units.GeV, aeff / units.m2, color="C1", label=r"$IC ~ (GFU): ~ \delta \approx 19.8^\circ, ~ \nu_\mu \bar{\nu}_\mu$", lw=5)


def plot_IC_GFU(ax, dt):
    energy, aeff, _ = get_ic_aeff_boat()

    fluence_limit = fluxes.get_limit_from_aeff(energy, aeff, livetime=dt) * energy ** 2
    p2, = ax.plot(energy / units.PeV, fluence_limit / (units.GeV * units.cm ** -2 * units.second ** -1),
                  color="C1", ls="--", lw=2, alpha=0.7, label=r"IC")


def plot_ic(ax, fluence=True):
    # data from publication

    # ------------- GFU
    e0 = 100 * units.TeV
    gfu_1hr_t0_2hr = [{"gamma": 1.5, "norm": 0.0359 * (units.GeV * units.cm ** -2), "e_min": 6.8 * units.TeV, "e_max": 9.9 * units.PeV},
                      {"gamma": 2.0, "norm": 0.0393 * (units.GeV * units.cm ** -2), "e_min": 0.83 * units.TeV, "e_max": 0.96 * units.PeV},
                      {"gamma": 2.5, "norm": 0.0143 * (units.GeV * units.cm ** -2), "e_min": 0.23 * units.TeV, "e_max": 0.086 * units.PeV}]

    for ele in gfu_1hr_t0_2hr:
        energies = np.geomspace(ele["e_min"], ele["e_max"])

        limit = ele["norm"] * (energies / e0) ** (2 - ele["gamma"])
        lw = 2
        alpha = 0.8
        if ele["gamma"] == 2:
            lw = 3
            alpha = 1
        p1, = ax.plot(energies / units.GeV, limit / (units.GeV * units.cm ** -2), color="C1", lw=lw, alpha=alpha)
        ax.text(1.1 * energies[-1] / units.GeV, limit[-1] / (units.GeV * units.cm ** -2), ele["gamma"], color="C1")

    # ------------- GRECCO
    grecco_t90 = [{"gamma": 1.5, "norm": 1.052 * (units.GeV * units.cm ** -2), "e_min": 30 * units.GeV, "e_max": 1.5 * units.TeV},
                  {"gamma": 2.0, "norm": 1.015 * (units.GeV * units.cm ** -2), "e_min": 26 * units.GeV, "e_max": 1.2 * units.TeV},
                  {"gamma": 2.5, "norm": 0.561 * (units.GeV * units.cm ** -2), "e_min": 15 * units.GeV, "e_max": 0.7 * units.TeV}]

    e0 = 1 * units.TeV
    for ele in grecco_t90:
        energies = np.geomspace(ele["e_min"], ele["e_max"])

        limit = ele["norm"] * (energies / e0) ** (2 - ele["gamma"])

        lw = 2
        alpha = 0.8
        if ele["gamma"] == 2:
            lw = 3
            alpha = 1

        p2, = ax.plot(energies / units.GeV, limit / (units.GeV * units.cm ** -2), color="C2", lw=lw, alpha=alpha)
        ax.text(0.25 * energies[0] / units.GeV, limit[0] / (units.GeV * units.cm ** -2), ele["gamma"], color="C2")
        # if ele["gamma"] == 2:
        #     ax.arrow(np.median(energies) / units.GeV, limit[0] / (units.GeV * units.cm ** -2), 0,
        #              - 0.9 * limit[0] / (units.GeV * units.cm ** -2), color="C2", lw=3,
        #              width=0.08)

    # ------------- ELOWEN
    elowen_t90 = [{"gamma": 2.0, "norm": 5.3e3 * (units.GeV * units.cm ** -2), "e_min": 0.5 * units.GeV, "e_max": 5 * units.GeV},
                  {"gamma": 2.5, "norm": 7.9e3 * (units.GeV * units.cm ** -2), "e_min": 0.5 * units.GeV, "e_max": 5 * units.GeV}]

    e0 = 1 * units.GeV
    for ele in elowen_t90:
        energies = np.geomspace(ele["e_min"], ele["e_max"])

        limit = ele["norm"] * (energies / e0) ** (2 - ele["gamma"])
        lw = 2
        alpha = 0.8
        if ele["gamma"] == 2:
            lw = 3
            alpha = 1
        p3, = ax.plot(energies / units.GeV, limit / (units.GeV * units.cm ** -2), color="C4", lw=lw, alpha=alpha)
        ax.text(1.1 * energies[-1] / units.GeV, limit[-1] / (units.GeV * units.cm ** -2), ele["gamma"], color="C4")

    ax2 = ax.twinx()
    ax2.set_axis_off()
    ax2.legend([p1, p2, p3], ["FRA, T0 [-1, +2] hr", "GRECO, T90", "ELOWEN, T0 +- 500s"], title="IceCube limits")


def plot_limit_boat():

    time_boat = Time('2020-10-09 13:16:59', scale="ut1")
    boat = SkyCoord.from_name('GRB221009A')

    sc = SensitivityCalculator(plot=False, zenith_limits=np.deg2rad([45, 95]), min_eff_area=None)
    t_obs, a_eff_avg, energies, zeniths = sc.get_observation_time_and_effective_area_energy(boat, time_boat, time_window=[- 1, 2] * u.h)
    energies *= units.eV
    a_eff = interpolate.interp1d(energies, a_eff_avg)

    dt = 3 * units.hour

    fig, ax = plt.subplots()

    opt, e, flux_limit = calculate_fluence_limit(a_eff, dt, 1e16, 1e20, differential=False)
    ax.plot(e / units.GeV, flux_limit / (units.GeV * units.cm ** -2), ls="--", label="RNO-G sensitivity\n(no background)", color="C3")
    ax.text(1.1 * e[-1] / units.GeV, flux_limit[-1] / (units.GeV * units.cm ** -2), "2.0", color="C3")

    opt, e, flux_limit = calculate_fluence_limit(a_eff, dt, 1e16, 1e20, differential=False, gamma=2.5)
    ax.plot(e / units.GeV, flux_limit / (units.GeV * units.cm ** -2), ls="--", lw=2, color="C3")
    ax.text(1.1 * e[-1] / units.GeV, flux_limit[-1] / (units.GeV * units.cm ** -2), "2.5", color="C3")

    opt, e, flux_limit = calculate_fluence_limit(a_eff, dt, 1e16, 1e20, differential=False, gamma=1.5)
    ax.plot(e / units.GeV, flux_limit / (units.GeV * units.cm ** -2), ls="--", lw=2, color="C3")
    ax.text(1.1 * e[-1] / units.GeV, flux_limit[-1] / (units.GeV * units.cm ** -2), "1.5", color="C3")


    plot_ic(ax)


    if 1:
        energy, aeff, a_eff = get_ic_aeff_boat()
        opt, e, flux_limit = calculate_fluence_limit(a_eff, dt, 0.83 * units.TeV, 0.96 * units.PeV)
        ax.plot(e / units.GeV, flux_limit / (units.GeV * units.cm ** -2), lw=5, ls="--", label="IC - recalculated", color="C1")

    if 0:
        draw_limit_differential(ax, sc)

    ax.set_xlabel(r"$E_\nu ~ /~ GeV$")
    ax.set_ylabel(r"$E^2 F(E) ~ / ~GeV ~cm^{-2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_ylim(1e-3, 1e3)
    ax.legend(loc="upper right", bbox_to_anchor=(0.999, 0.75))
    ax.grid()

    fig.tight_layout()
    plt.savefig("fluence_limit_boat.png", transparent=True)


def draw_limit_differential(ax, sc):

    t_obs, a_eff_avg, energies, zeniths = sc.get_observation_time_and_effective_area_energy(boat, time_boat, time_window=[- 1, 2] * u.h)
    energies *= units.eV

    fluence_limit = fluxes.get_limit_from_aeff(
        energies, a_eff_avg, livetime=1, upperLimOnEvents=2.3) * energies ** 2 / (units.GeV * units.cm ** -2)
    ax.plot(energies / units.GeV, fluence_limit, ":", color="C3", label="RNO-G sensitivity")

    energy, aeff, a_eff = get_ic_aeff_boat()

    fluence_limit = fluxes.get_limit_from_aeff(
        energy, aeff, livetime=1, upperLimOnEvents=2.3) * energy ** 2 / (units.GeV * units.cm ** -2)
    ax.plot(energy / units.GeV, fluence_limit, ls=":", color="C1", label="IC GFU limit")


def plot_limit_differential():
    time_boat = Time('2020-10-09 13:16:59', scale="ut1")
    boat = SkyCoord.from_name('GRB221009A')

    sc = SensitivityCalculator(plot=False, zenith_limits=np.deg2rad([45, 95]), min_eff_area=None)
    t_obs, a_eff_avg, energies, zeniths = sc.get_observation_time_and_effective_area_energy(boat, time_boat, time_window=[- 1, 2] * u.h)
    energies *= units.eV

    fig, ax = plt.subplots()

    draw_limit_differential(ax, sc)

    if 0:
        # Cross-check
        a_eff = interpolate.interp1d(energies, a_eff_avg)
        dt = 3 * units.hour
        opt, e, flux_limit = calculate_fluence_limit(a_eff, dt, 1e16, 1e20, differential=True)
        ax.plot(e / units.GeV, flux_limit / (units.GeV * units.cm ** -2), ls="-", lw=1, label="differential", color="C3")

    ax.set_xlabel(r"$E_\nu ~ /~ GeV$")
    ax.set_ylabel(r"$E^2 F(E) ~ / ~GeV ~cm^{-2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()

    fig.tight_layout()
    plt.savefig("fluence_limit_boat_differential.png", transparent=True)


if __name__ == "__main__":

    sc = SensitivityCalculator(plot=True, zenith_limits=np.deg2rad([45, 95]), min_eff_area=None)
    time_boat = Time('2020-10-09 13:16:59', scale="ut1")

    boat = SkyCoord.from_name('GRB221009A')

    if 1:
        fig, ax = plt.subplots()

        sc.plot_sky_corr_in_altaz(ax, boat, time_boat, [- 12 , 12] * u.h, time_scale="hr", title="")
        # ax.set_xlim(-10, 370)
        ax.legend()
        fig.tight_layout()
        plt.savefig("boat.png", transparent=True)

    if 1:
        fig, ax = plt.subplots()
        # T0 -1hr +2hr = 80.5 - 67.9
        sc.plot_effective_area(ax, boat, time_boat, time_window=[- 1 , 2] * u.h,
                               label=r"$RNO-G: ~ \theta \in [67.9^\circ, 80.5^\circ], ~ all ~ flavors$", p_kwargs=dict(color="C3", ls="--", lw=5))
        plot_ic_aeff(ax)

        ax.legend(ncols=1)
        fig.tight_layout()
        plt.savefig("effective_area_boat.png", transparent=True)

    if 1:
        plot_limit_boat()
        plot_limit_differential()
