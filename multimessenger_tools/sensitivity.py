import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import json
import pandas as pd
import os

from scipy import interpolate, integrate, optimize
from itertools import groupby

from NuRadioReco.utilities import units
from NuRadioMC.utilities import cross_sections, fluxes


DIRECTORY = os.path.dirname(os.path.realpath(__file__))


def calculate_fluence_limit(a_eff, dt, e_min, e_max, gamma=2, num_events=2.3, differential=False):
    """
    Calculate a sensitivity limit for the neutrino fluence within a certain time window.

    Parameters
    ----------

    a_eff: callable
        Returns the effective area as a function of the energy (in eV).

    dt: float
        Time interval for which to calculate the fluence limit

    e_min: float
        Lower energy boundary

    e_max: float
        Upper energy boundary

    gamma: float
        Fixed spectral index. Only relevant if `differential == False`. (Default: 2)

    num_events: float
        Number of events defining the sensitivity. (Default: 2.3)

    differential: bool
        If True, calculate the differential limit in full decadale energy bins (each half decade). A E^-1 box spectrum is assumed in each energy bin.
        Otherwise assume a single power law spectrum with the spectral index `gamma` is assumed. (Default: False)

    Returns
    -------

    opt: float
        The flux normalisation which describes the flux limit.

    e: array of floats
        The (central) energy values for the flux limit.

    int_flux: array of floats
        The flux limit times the time interval times the energy to the power of 2.
    """

    # flux model
    def flux(energy, normalization, gamma=2, e0=100 * units.TeV):
        return normalization * (energy / e0) ** -gamma * 1e-18 * (units.GeV ** -1 * units.cm ** -2 * units.second ** -1 * units.sr ** -1)

    # numerical integration over flux * a_eff * dt in energy
    def integrate_over_time_integrated_flux(norm, dt, a_eff, e_min, e_max, gamma=2):
        res = integrate.quad(lambda e: flux(e, norm, gamma=gamma) * dt * a_eff(e), e_min, e_max)
        return res[0]

    # Helper function for optimize.brentq
    def sensitivity_limit(norm, dt, a_eff, e_min, e_max, gamma, num_events):
        return integrate_over_time_integrated_flux(norm, dt, a_eff, e_min, e_max, gamma=gamma) - num_events

    if not differential:
        # Power law limit
        opt = optimize.brentq(sensitivity_limit, 1e-2, 1e8, args=(dt, a_eff, e_min, e_max, gamma, num_events))
        e = np.geomspace(e_min, e_max)

        int_flux = flux(e * units.eV, opt, gamma=gamma) * (e * units.eV) ** 2 * dt
        return opt, e, int_flux

    else:
        # Piece-wise / differential limit
        n_decades = np.log10(e_max / e_min)
        if n_decades % 1 != 0:
            raise ValueError()

        energy_bins = np.log10(e_min) + np.arange(0, n_decades + 1, 0.5)
        int_flux = []
        e = []
        opt = []
        for e_min2 in energy_bins[:-1]:
            e_max2 = e_min2 + 1
            if e_max2 > np.log10(e_max):
                continue

            opt_dec, _, int_flux_dec = calculate_fluence_limit(a_eff, dt, 10 ** e_min2, 10 ** e_max2, gamma=1, num_events=num_events, differential=False)
            opt.append(opt_dec)
            int_flux.append(np.median(int_flux_dec))
            e.append(10 ** (e_min2 + 0.5))

        return np.array(opt), np.array(e), np.array(int_flux)


def draw_zenith_angles_on_top_xaxis(
        ax, theta=np.deg2rad([95, 75, 60, 45]), fontsize="medium"):
    """
    Draws a second x-axis at the top of the figure which shows
    the theta values which corrspond to the cos(theta) values from the bottom
    x-axis

    Paramters
    ---------

    ax: `matplotlib.pyplot.axis`
        The axis to plot a top x-axis on

    theta: list/array of floats
        Zenith angle values to draw (in radiants). (Default: np.deg2rad([95, 75, 60, 45]))

    fontsize: str or int
        Fontsize of the axis labels. (Default: "medium")
    """
    ax2 = ax.twiny()

    ax2.set_xticks(np.cos(theta))
    ax2.set_xticklabels(
        [r"%.1f$^\circ$" % x for x in np.rad2deg(theta)], fontsize=fontsize)

    ax2.set_xlim(ax.get_xlim())


def get_source_zenith(sky_corr, times, earth_location):
    """
    Calculate zenith angle of a source in the sky for an
    experiment on earth of a specific timestamp.

    Parameters
    ----------

    sky_corr: astropy.coordinates.SkyCoord
        Location of a source in the sky

    time: astropy.time.Time or list of that
        Timestamps for which to calculate the zenith angle

    earth_location: astropy.coordinates.EarthLocation
        Location of the experiment on earth

    Returns
    -------

    zenith: np.ndarray
        The zenith angle under which the source is visible.
    """
    return np.array(np.pi / 2 - sky_corr.transform_to(
        AltAz(obstime=times, location=earth_location)).alt / u.rad, dtype=float)


def is_in_fov(sky_corr, times, earth_location, fov):
    """ Calclate whether a source is within the FOV (field-of-fiew) of an experiment

    Parameters
    ----------

    sky_corr: astropy.coordinates.SkyCoord
        Location of a source in the sky

    times: stropy.time.Time (or an array of that)
        Timestamps for which to calculate the FOV

    earth_location: astropy.coordinates.EarthLocation
        Location of the experiment on earth

    fov: tuple of floats or callable
        Defines the FOV of an experiment (typically defined as the minimal and maximal
        zenith angle a experiment is sensitve to).

            * Tuple of floats: Interpreted as the minumum and maximum
              zenith angle a experiemnt is sensitive to (zenith_min, zenith_max)
            * Callable: Expects a function which takes as argument the azimuth of
              the source and returns two floats: (zenith_min, zenith_max)

    Returns
    -------

    is_in_fov: np.ndarray(bool)
        Return a bool for each timestamp whether its in the FOV
    """

    if isinstance(times, Time):
        times = [times]

    if callable(fov):
        #NOTE / BUG different azimuth definition between RNOG and AltAz
        is_in_fov = []
        for time in times:
            altaz = sky_corr.transform_to(AltAz(obstime=time, location=earth_location))
            zen_min, zen_max = fov(altaz.az / u.deg)
            is_in_fov.append(zen_min < float(np.pi / 2 - altaz.alt / u.rad) < zen_max)

        is_in_fov = np.array(is_in_fov, dtype=bool)
    else:
        soruce_zeniths = np.array([get_source_zenith(sky_corr, t, earth_location) for t in times])
        zen_min, zen_max = fov
        is_in_fov = np.all([zen_min < soruce_zeniths, soruce_zeniths < zen_max], axis=0)

    return np.squeeze(is_in_fov)


def get_effective_area(path=f"{DIRECTORY}/data/Veff_fine_20.json", energy=1.1e18, plot=False):
    """
    Parse json file to get the effective area as a funcion of the zenith angle
    of RNO-G for a specific energy.
    """
    df = pd.DataFrame(json.load(open(path, "r")))
    energies = np.unique(df.energy)

    idx = np.argmin(np.abs(energies - energy))
    if energies[idx] < energy:
        energy_low = energies[idx]
        energy_up = energies[idx + 1]
    else:
        energy_low = energies[idx - 1]
        energy_up = energies[idx]

    selected_df = df[df.energy == energy_low]
    veffs_low = np.array([row.veff["2.00sigma"][0] for _, row in selected_df.iterrows()])

    selected_df = df[df.energy == energy_up]
    veffs_up = np.array([row.veff["2.00sigma"][0] for _, row in selected_df.iterrows()])

    veffs = veffs_low + (veffs_up - veffs_low) * (energy - energy_low) / (energy_up - energy_low)

    aeff = veffs / cross_sections.get_interaction_length(energy)

    if plot:
        figx, ax = plt.subplots()
        ax.set_title(f"Energy {energy:.2e}")
        theta = (selected_df.thetamin + selected_df.thetamax) / 2
        ax.plot(np.cos(theta), aeff / units.m ** 2)

        # ax.set_ylim(0, 0.036)
        ax.grid()
        ax.set_xlabel("$cos(zenith)$")
        ax.set_ylabel(r"$effective \, area \, / \, m^2$")
        figx.tight_layout()
        figx.savefig(f"effective_area_cos_zen_{energy}.png")

    return selected_df.thetamin, selected_df.thetamax, aeff, energy


def get_effective_area_function(path=f"{DIRECTORY}/data/Veff_fine_20.json", energy=1.1e18, norm=False):
    """ Returns the effective area as a scipy.interpolate.interp1d function """
    thetamin, thetamax, aeff, energy = get_effective_area(path, energy)
    theta = np.array((thetamin + thetamax) / 2)

    sort = np.argsort(theta)
    theta = theta[sort]
    aeff = aeff[sort]

    if norm:
        aeff = aeff / np.amax(aeff)

    return interpolate.interp1d(theta, aeff, bounds_error=False, fill_value=0), energy


def get_effective_area_energy_function(path=f"{DIRECTORY}/data/Veff_fine_20.json", norm=False, plot=False):
    """ Returns a 2d interpolation function f(energy, theta) of the effective area """
    df = pd.DataFrame(json.load(open(path, "r")))
    energies = np.unique(df.energy)

    aeffs = []
    energies_sel = []
    for energy in energies:
        selected_df = df[df.energy == energy]

        veffs = np.array([row.veff["2.00sigma"][0] for _, row in selected_df.iterrows() if "2.00sigma" in row.veff])

        if len(veffs) == 0:
            # print(np.log10(energy))
            continue

        energies_sel.append(energy)
        theta = np.array((selected_df.thetamin + selected_df.thetamax) / 2)

        sort = np.argsort(theta)
        theta = theta[sort]
        veffs = veffs[sort]

        aeff = veffs / cross_sections.get_interaction_length(energy)
        aeffs.append(aeff)

    aeffs = np.array(aeffs)
    energies_sel = np.array(energies_sel)

    if norm:
        aeffs = aeffs / np.amax(aeffs)

    if plot:
        fig, ax = plt.subplots()

        pcm = ax.pcolormesh(np.cos(theta), np.log10(energies_sel), aeffs / units.m2, shading='nearest',
                            norm=colors.LogNorm(vmin=500, vmax=aeffs.max() / units.m2))

        ax.grid()

        cbi = plt.colorbar(pcm, pad=0.02)
        ax.set_ylabel(r"$log(E_\nu / eV)$")
        ax.set_xlabel(r"$cos(\theta)$")
        cbi.set_label(r"$effective~area~/~m^2$")
        fig.tight_layout()
        plt.savefig("effective_area_2d.png")

    # add right bin edge of last bin. This allows to plot sensitivity until 1e20
    energies_sel = np.hstack([energies_sel, 1e20])
    aeffs = np.vstack([aeffs, aeffs[-1]])

    return interpolate.RegularGridInterpolator((energies_sel, theta), aeffs, bounds_error=False, fill_value=0)


def plot_sky_corr_in_altaz(
        ax, sky_corr, earth_location, time, time_window, time_scale="hr",
        zenith_limits=None, draw_eff_area=True, title=None, dt=10 * u.min, plot_alert=True):
    """
    Plot the location of a source in the sky in the local corrdinate system of a experiment
    within the corrdinate system (azimuth, zenith)
    """

    if not isinstance(time_window, u.quantity.Quantity):
        raise ValueError("Expects \"time_window\" to have a astropy unit "
                         "(i.e., be of type astropy.unit.quantity.Quantity)")

    if not isinstance(dt, u.quantity.Quantity):
        raise ValueError("Expects \"dt\" to have a astropy unit "
                         "(i.e., be of type astropy.unit.quantity.Quantity)")

    if time_window.isscalar:
        time_range =  np.arange(- time_window / 2 / u.s,
                                (time_window / 2 + dt) / u.s, dt / u.s) * u.s
    elif len(time_window) == 2:
        time_range = np.arange(time_window[0] / u.s,
                                (time_window[1] + dt) / u.s, dt / u.s) * u.s
    else:
        raise ValueError(f"Expects \"time_window\" to be a scalar or two scalars, is {time_window}")

    # get bin centers
    time_range = time_range[:-1] + np.diff(time_range) / 2

    data = []
    for t_add in time_range:
        altaz = sky_corr.transform_to(AltAz(obstime=time + t_add, location=earth_location))
        data.append([altaz.az / u.deg, altaz.alt / u.deg, t_add.to_value(time_scale)])
    data = np.array(data)

    cmap = "twilight"
    if draw_eff_area:
        cmap += "_shifted"

    sct = ax.scatter(data[:, 0], 90 - data[:, 1] , c=data[:, 2], marker="o", s=15, cmap=cmap, zorder=10, edgecolor="k", lw=0.1)  # cmap=hsv
    cbi = plt.colorbar(sct, pad=0.02)
    if time_scale == "s":
        time_scale = "seconds"
    cbi.set_label(f"{time_scale} since alert")
    # cbi.set_label(f"{time_scale} since {time}")

    ax.set_xlabel("azimuth / deg")
    ax.set_ylabel("zenith / deg")

    if draw_eff_area:
        theta_min, theta_max, aeff = get_effective_area()

        aeff_norm = aeff / np.amax(aeff)
        ax.set_ylim(ax.get_ylim())
        for a, t1, t2 in zip(aeff_norm, theta_min, theta_max):
            ax.axhspan(np.rad2deg(t1), np.rad2deg(t2), color="k", alpha=(1-a))

    if zenith_limits is not None:
        l = "FOV limit"
        for zen in zenith_limits:
            ax.axhline(np.rad2deg(zen), ls="--", color="k", label=l)
            l = ""

        if not draw_eff_area:
            y1, y2 = ax.get_ylim()
            ax.set_ylim(y1, y2)
            ax.axhspan(y1, np.rad2deg(zenith_limits[0]), color="gray", alpha=0.8)
            ax.axhspan(np.rad2deg(zenith_limits[1]), y2, color="gray", alpha=0.8)

    if title is not None:
        ax.set_title(title)

    if plot_alert:
        altaz = sky_corr.transform_to(AltAz(obstime=time, location=earth_location))
        ax.plot(altaz.az / u.deg, 90 - altaz.alt / u.deg, "C1*", markersize=12, zorder=10)

    ax.grid()


def plot_effective_area_1d(ax):
    pass


def transfrom_local_to_equitorial(location, zenith,
                                  azimuth=np.linspace(-180, 180, 100, endpoint=False) * u.deg,
                                  time=Time('2024-04-20 12:00:00', scale="ut1")):
    """
    Transforms local coordinates (zenith, azimuth) at a given location and time
    into equatorial coordinates (right ascension, declination).

    Parameters
    ----------

    location: `astropy.coordinates.EarthLocation`
        Specify the location at Earth associated to the (zenith, azimuth) coordinates.

    zenith: `astropy.unit.quantity.Quantity`
        Specify the zenith angle of the coordinate.

    azimuth: `astropy.unit.quantity.Quantity`
        Specify the azimuth angle of the coordinate. Should be a list / array of angles.
        (Default: `np.linspace(-180, 180, 100, endpoint=False) * u.deg`)

    time: `astropy.time.Time`
        Time of the local coordinate.

    Returns
    -------

    ra: np.array
        Right ascension of all local coordinates.

    dec: np.array
        Declination of all local coordinates.
    """

    if not isinstance(zenith, u.quantity.Quantity):
        raise ValueError("Expects \"zenith\" to have a astropy unit "
                         "(i.e., be of type astropy.unit.quantity.Quantity)")

    if not isinstance(azimuth, u.quantity.Quantity):
        raise ValueError("Expects \"azimuth\" to have a astropy unit "
                         "(i.e., be of type astropy.unit.quantity.Quantity)")

    ra = np.zeros(len(azimuth))
    dec = np.zeros(len(azimuth))
    for idx, azi in enumerate(azimuth):
        # 180 is south
        # alt angle is elevation angle (90 - zenith)
        pos_aa = AltAz(az=azi, alt=90 * u.deg - zenith, obstime=time, location=location)
        c = SkyCoord(pos_aa)
        ra[idx] = c.transform_to('icrs').ra / u.deg
        dec[idx] = c.transform_to('icrs').dec / u.deg

    return ra, dec


class SensitivityCalculator:

    def __init__(self, min_eff_area=0.001 * units.km2, zenith_limits=np.deg2rad([55, 90]), plot=False):
        """
        The `zenith_limits` define the field-of-view (FOV) of the experiment. Can be set or calculated from
        the effective area (see explanation of the parameters).

        Parameters
        ----------

        min_eff_area: float
            Specifies the minimum effective area (a_eff) to define the field-of-view (FOV)
            of the experiment: If a_eff(theta) > a_eff_min, theta is in FOV. (Default: 0.005 * units.km2)

        zenith_limits: tuple of floats
            Define the minimum and maximum zenith angle which is in the FOV.
            Only used if "min_eff_area = None". (Default: np.deg2rad([55, 90]))

        """
        # TODO: Use more accurate numbers
        self.summit_station = EarthLocation(lat=72.5 * u.deg, lon=-38.5 * u.deg, height=2800 * u.m)
        self._f_aeff = None

        if min_eff_area is not None or plot:
            test_zeniths = np.linspace(0, np.pi, 200)
            f_aeff, energy = get_effective_area_function(energy=1.1e18)
            aeffs = f_aeff(test_zeniths)

        if min_eff_area is not None:

            # Make sure that the aeffs are well behaved, i.e., 0, 0, ... 0, >0, >0, ..., >0, 0, 0, ..., 0 (0 = min_eff_area)
            in_fov = aeffs > min_eff_area
            groups = [i for i, _ in groupby(in_fov)]
            if len(groups) != 3:
                raise ValueError("The effective area is not well behaved. Stop!")
            self.zenith_limits = [test_zeniths[in_fov].min(), test_zeniths[in_fov].max()]

        else:
            self.zenith_limits = zenith_limits

        print(f"Define FOV between {np.around(np.rad2deg(self.zenith_limits), 2)} deg")

        if plot:
            fig, ax = plt.subplots()
            ax.plot(np.cos(test_zeniths), aeffs / units.m2, color="C1")
            x1, x2 = ax.get_xlim()

            if min_eff_area is not None:
                label = rf"$FOV: ~ a_{{eff}}(\theta) > {min_eff_area / units.m2} m^2$"
                fname = (f"eff_area_zenith_{energy:.1e}eV_{np.rad2deg(self.zenith_limits[0]):.1f}_"
                         f"{np.rad2deg(self.zenith_limits[1]):.1f}.png")
            else:
                label = "FOV"
                fname = (f"eff_area_zenith_{np.rad2deg(self.zenith_limits[0]):.1f}_"
                         f"{np.rad2deg(self.zenith_limits[1]):.1f}.png")

            ax.axvline(np.cos(self.zenith_limits[0]), color="k", ls="--", label=label)
            ax.axvspan(np.cos(self.zenith_limits[0]), x2, color="gray", alpha=0.8)
            ax.axvline(np.cos(self.zenith_limits[1]), color="k", ls="--")
            ax.axvspan(x1, np.cos(self.zenith_limits[1]), color="gray", alpha=0.8)
            ax.set_xlim(x1, x2)

            draw_zenith_angles_on_top_xaxis(ax)

            ax.set_xlabel(r"$cos(\theta)$")
            ax.set_ylabel(r"$eff. ~ area ~ / ~ m^2$")
            ax.legend(title=fr"$E_\nu = {energy:.2e} ~ eV$")
            ax.grid()
            fig.tight_layout()
            plt.savefig(fname)

    def get_effective_area_function(self, **kwargs):
        """ Wrapper around get_effective_area_function, caches function """
        if self._f_aeff is None:
           self._f_aeff = get_effective_area_function(**kwargs)
        return self._f_aeff

    def transfrom_local_to_equitorial(
            self, zenith, time=Time('2024-04-20 12:00:00', scale="ut1"),
            azimuth=np.linspace(-180, 180, 100, endpoint=False) * u.deg):
        """ Wrapper around transfrom_local_to_equitorial """
        return transfrom_local_to_equitorial(self.summit_station, zenith, time, azimuth)

    def is_in_fov(self, sky_corr, times):
        """ Wrapper around is_in_fov """
        return is_in_fov(sky_corr, times, self.summit_station, fov=self.zenith_limits)

    def get_source_zenith(self, sky_corr, time):
        """ Wrapper around get_source_zenith """
        return get_source_zenith(sky_corr, time, self.summit_station)

    def get_observation_time_and_effective_area(
            self, sky_corr, time, time_window=24 * u.hour,
            dt=1 * u.min, average_over_fov=False):
        """
        Calculate the observation time, i.e., the time in which the source is within the FOV and the
        average effective area for this time. Both are calculated within a given time range.

        Parameters
        ----------
        sky_corr: astropy.coordinates.SkyCoord
            Location of a source in the sky

        time: astropy.time.Time
            Timestamps of the alert

        time_window: astropy.units.quantity.Quantity
            Time window relative to the alert time.

        dt: astropy.units.quantity.Quantity
            Defines time interval after which to calculate if source is in FOV and effective area

        average_over_fov: bool
            Calculate average of effective area (a_eff) of entrie time window or only when in FOV.

                * If True: Average a_eff only when source is in FOV (within t_obs).
                * If False (default): Average over entire time window even if not in FOV (which enters as 0)

        Returns
        -------

        t_obs: astropy.units.quantity.Quantity
            Total observation time, i.e., time the source is within the FOV, in seconds

        a_eff_avg: float
            Average effective area of the source during observation
        """
        if not isinstance(time_window, u.quantity.Quantity):
            raise ValueError("Expects \"time_window\" to have a astropy unit "
                             "(i.e., be of type astropy.unit.quantity.Quantity)")

        if not isinstance(dt, u.quantity.Quantity):
            raise ValueError("Expects \"dt\" to have a astropy unit "
                             "(i.e., be of type astropy.unit.quantity.Quantity)")

        if time_window.isscalar:
            time_range =  np.arange(- time_window / 2 / u.s,
                                    (time_window / 2 + dt) / u.s, dt / u.s) * u.s
        elif len(time_window) == 2:
            time_range = np.arange(time_window[0] / u.s,
                                   (time_window[1] + dt) / u.s, dt / u.s) * u.s
        else:
            raise ValueError(f"Expects \"time_window\" to be a scalar or two scalars, is {time_window}")

        # get bin centers
        time_range = time_range[:-1] + np.diff(time_range) / 2

        times = time + time_range
        zeniths = self.get_source_zenith(sky_corr, times)

        zen_min, zen_max = self.zenith_limits
        in_fov = np.all([zen_min < zeniths, zeniths < zen_max], axis=0)

        t_obs = dt * np.sum(in_fov)

        if average_over_fov:
            norm = np.sum(in_fov)
        else:
            norm = len(in_fov)

        f_aeff, _ = get_effective_area_function()
        a_eff_avg = np.sum([f_aeff(theta) for theta in zeniths[in_fov]]) / norm

        return t_obs, a_eff_avg


    def get_observation_time_and_effective_area_energy(
            self, sky_corr, time, time_window=24 * u.hour,
            dt=1 * u.min):
        """
        Calculate the observation time `t_obs`, i.e., the time in which the source is within the FOV, and the
        average effective area `a_eff_avg` within `t_obs` (NOT within the specified `time_window`).
        `time_window` specifies in which time window / period `t_obs` and `a_eff_avg` are calculated.

        Parameters
        ----------
        sky_corr: astropy.coordinates.SkyCoord
            Location of a source in the sky

        time: astropy.time.Time
            Timestamps of the alert

        time_window: astropy.units.quantity.Quantity
            Time window relative to the alert time.

        dt: astropy.units.quantity.Quantity
            Defines time interval after which to calculate if source is in FOV and effective area

        Returns
        -------

        t_obs: astropy.units.quantity.Quantity
            Total observation time, i.e., time the source is within the FOV, in seconds

        a_eff_avg: float
            Average effective area of the source during observation
        """
        if not isinstance(time_window, u.quantity.Quantity):
            raise ValueError("Expects \"time_window\" to have a astropy unit "
                             "(i.e., be of type astropy.unit.quantity.Quantity)")

        if not isinstance(dt, u.quantity.Quantity):
            raise ValueError("Expects \"dt\" to have a astropy unit "
                             "(i.e., be of type astropy.unit.quantity.Quantity)")

        if time_window.isscalar:
            time_range =  np.arange(- time_window / 2 / u.s,
                                    (time_window / 2 + dt) / u.s, dt / u.s) * u.s
        elif len(time_window) == 2:
            time_range = np.arange(time_window[0] / u.s,
                                   (time_window[1] + dt) / u.s, dt / u.s) * u.s
        else:
            raise ValueError(f"Expects \"time_window\" to be a scalar or two scalars, is {time_window}")

        # get bin centers
        time_range = time_range[:-1] + np.diff(time_range) / 2

        times = time + time_range
        zeniths = self.get_source_zenith(sky_corr, times)

        zen_min, zen_max = self.zenith_limits
        in_fov = np.all([zen_min < zeniths, zeniths < zen_max], axis=0)

        t_obs = dt * np.sum(in_fov)

        energies = 10 ** np.arange(16, 20.1, 0.5)

        f_aeff = get_effective_area_energy_function()

        a_eff_avg = np.array([np.sum([f_aeff((e, theta)) for theta in zeniths[in_fov]]) / np.sum(in_fov) for e in energies])

        return t_obs, a_eff_avg, energies, zeniths[in_fov]


    def plot_effective_area(self, ax, sky_corr, time, time_window=24 * u.hour, label="", p_kwargs={}):
        t_obs, a_eff_avg, energies, zeniths = self.get_observation_time_and_effective_area_energy(sky_corr, time, time_window=time_window)
        energies *= units.eV

        ax.plot(energies / units.GeV, a_eff_avg / units.m2, label=label, **p_kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_\nu ~ / ~ GeV$")
        ax.set_ylabel(r"$eff.~area~/~m^2$")
        ax.grid()


    def plot_sensitivity(self, ax, sky_corr, time, number_of_days=1, time_window=24 * u.hour, label="", kind="flux", p_kwargs={}):

        t_obs, a_eff_avg, energies, zeniths = self.get_observation_time_and_effective_area_energy(sky_corr, time, time_window=time_window)
        energies *= units.eV
        t_obs_tot_astro = number_of_days * t_obs
        t_obs_tot = float(t_obs_tot_astro / u.s) * units.second

        if kind == "flux":
            flux_limit = fluxes.get_limit_from_aeff(energies, a_eff_avg, livetime=t_obs_tot) * energies ** 2
            flux_limit = flux_limit / (units.TeV * units.cm ** -2 * units.second ** -1)
            ax.plot(energies / units.GeV, flux_limit, label=label, **p_kwargs)
            ax.set_ylabel(r"$E_\nu^2 ~ \Phi_{\nu + \bar{\nu}} ~ / ~ TeV ~ cm^{-2} ~ s^{-1}$")

        # I do not think this is actually correct
        # elif kind == "fluence":

        #     fluence_limit = fluxes.get_limit_from_aeff(
        #         energies, a_eff_avg, livetime=t_obs_tot) * energies ** 2 / (units.GeV * units.cm ** -2 * units.second ** -1)

        #     ax.plot(energies / units.GeV, fluence_limit, label=label, **p_kwargs)
        #     ax.set_ylabel(r"$E_\nu^2 ~ \Phi_{\nu + \bar{\nu}} ~ \cdot ~ \Delta T  ~ / ~ GeV ~ cm^{-2}$")

        else:
            raise ValueError(f"Unknown type for \"kind\": {kind} (valid are: \"flux\")")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_\nu ~ / ~ GeV$")
        ax.grid()
        ax.legend()


    def plot_sky_corr_in_altaz(self, ax, sky_corr, time, time_range, time_scale="hr", title=""):

        t_obs, a_eff_avg = self.get_observation_time_and_effective_area(sky_corr, time, time_range)
        t_obs = t_obs.to("hr")
        title += rf"$t_{{obs}} = {t_obs:.2f}, \langle a_{{eff}} \rangle = {a_eff_avg / units.km2:.4f} km^2$"
        return plot_sky_corr_in_altaz(ax, sky_corr, self.summit_station, time, time_range, time_scale,
                                      zenith_limits=self.zenith_limits, draw_eff_area=False, title=title)
