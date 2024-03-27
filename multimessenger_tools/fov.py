from multimessenger_tools import sensitivity

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import numpy as np
import itertools
import cmasher as cmr


def draw_source(ax, coor, fmt, s, va="bottom", color="darkgrey"):

    x = coor.ra / u.deg
    if x > 180:
        x -= 360

    dy = 5 if va == "top" else -5

    ha = "right"  # horizontal alignment
    dx = 5
    if x < 0:
        ha = "left"
        dx = -5

    ax.plot(np.deg2rad([x]), np.deg2rad([coor.dec / u.deg]), fmt)
    ax.text(np.deg2rad([x]), np.deg2rad([coor.dec / u.deg]), s,
            fontsize="x-small", ha=ha, va=va, color=color)


def draw_sources(ax):
    NGC1068 = SkyCoord('02h42m41s', '-0d00m48s', frame='icrs')
    draw_source(ax, NGC1068, "*b", "NGC 1068", va="top")

    TXS0506056 = SkyCoord('05h09m25s', '05d49m09s', frame='icrs')
    draw_source(ax,TXS0506056, "*b", "TXS 0506+056")

    # TA_hotspot = SkyCoord(146.7 * u.deg, 43.2 * u.deg, frame='icrs') #diameter of 30-40 deg.
    # draw_source(ax, TA_hotspot, "or", "TA Hotspot")

    GRB_221009A = SkyCoord(288.3 * u.deg, 19.7 * u.deg, frame='icrs')
    draw_source(ax, GRB_221009A, "*y", "GRB 221009A")


    PKS1424_240 = SkyCoord('14h27m00s', '23d47m40s', frame='icrs')
    draw_source(ax, PKS1424_240, "*b", "PKS 1424+240")

    # GB6_J1542_6129 = SkyCoord('15h42m56s', '61d29m55s', frame='icrs')
    # M87 = SkyCoord('12h30m49s', '12d23m28s', frame='icrs')
    # Markarian421 = SkyCoord('11h04m19s', '38d11m41s', frame='icrs')

    Markarian501 = SkyCoord('16h53m52s', '39d45m37s', frame='icrs')
    draw_source(ax, Markarian501, "*C1", "Mk 501")

    M82 = SkyCoord('09h55m52s', '69d40m47s', frame='icrs')
    draw_source(ax, M82, "*C1", "M82", va="top")


def add_instantaneous_bands(ax, time, zenith, sc, plot_kwargs):

    def get_ra_dec(zenith):
        ra, dec = sc.transfrom_local_to_equitorial(zenith * u.deg, time=time)

        ra[ra > 180] -= 360
        sort = np.argsort(ra)

        return np.deg2rad(ra[sort]), np.deg2rad(dec[sort])

    ra, dec = get_ra_dec(zenith)
    ax.plot(ra, dec, **plot_kwargs)



def draw_daily_averaged_fov(ax, sc, time=Time('2024-04-20 12:00:00', scale="ut1")):

    ras = np.linspace(-180, 180, 45)
    decs = np.linspace(-90, 90, 60)
    a_effs = []

    for r, d in itertools.product(ras, decs):
        if -30 < d < 75:
            coor = SkyCoord(ra=r * u.deg, dec=d * u.deg, frame='icrs')
            _, a_eff  = sc.get_observation_time_and_effective_area(coor, time, dt=10 * u.min)
            a_effs.append(a_eff)
        else:
            a_effs.append(0)

    a_eff = np.array(a_effs).reshape(len(ras), -1)

    pcm = ax.pcolormesh(np.deg2rad(ras), np.deg2rad(decs),
                        a_eff.T / 1000 ** 2, shading="gouraud", cmap=plt.get_cmap('cmr.freeze'))
    cbr = fig.colorbar(pcm, pad=0.02)
    cbr.set_label(r"daily avg. eff. area / km$^2$")

    ax.set_ylabel("declination / deg")

    ax.set_xticks([])
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi / 4))
    ax.grid(which="both")
    # ax.grid(axis="y")
    ax.set_yticks(np.deg2rad([-75, -60, -30, 0, 30, 60, 75]))
    ax.spines["geo"].set_linewidth(1)


def draw_instantaneous_fov(ax, sc, time=Time('2024-04-20 12:00:00', scale="ut1")):
    ras = np.linspace(-180, 180, 45)
    decs = np.linspace(-90, 90, 60)
    a_effs = []

    f_aeff, _ = sc.get_effective_area_function()

    for r, d in itertools.product(ras, decs):
        coor = SkyCoord(ra=r * u.deg, dec=d * u.deg, frame='icrs')
        zenith = sc.get_source_zenith(coor, time)
        a_effs.append(f_aeff(zenith))

    a_eff = np.array(a_effs).reshape(len(ras), -1)

    pcm = ax.pcolormesh(np.deg2rad(ras), np.deg2rad(decs),
                        a_eff.T / 1000 ** 2, shading="gouraud", cmap=plt.get_cmap('cmr.freeze'))
    cbr = fig.colorbar(pcm, pad=0.02)
    cbr.set_label(r"eff. area / km$^2$")

    ax.set_ylabel("declination / deg")

    ax.set_xticks([])
    ax.xaxis.set_minor_locator(MultipleLocator(np.pi / 4))
    ax.grid(which="both")
    # ax.grid(axis="y")
    ax.set_yticks(np.deg2rad([-75, -60, -30, 0, 30, 60, 75]))
    ax.spines["geo"].set_linewidth(1)

def plot_integrated_fov():
    sc = sensitivity.SensitivityCalculator()
    time = Time('2024-04-20 12:00:00', scale="ut1")

    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(projection='mollweide'))
    draw_daily_averaged_fov(ax, sc, time)

    draw_sources(ax)
    fig.tight_layout()

    plt.savefig("fov_integrated_x.png", transparent=False)

    add_instantaneous_bands(ax, time, 45, sc, dict(color="grey", lw=1))
    add_instantaneous_bands(ax, time, 95, sc, dict(color="grey", lw=1, label=r"45$^\circ$ - 95$^\circ$"))

    add_instantaneous_bands(ax, time, 55, sc, dict(color="grey", lw=2))
    add_instantaneous_bands(ax, time, 90, sc, dict(color="grey", lw=2, label=r"55$^\circ$ - 90$^\circ$"))

    ax.legend(title="Instantaneous FOV", ncols=2, loc='upper left', bbox_to_anchor=(0.1, 1.5))
    fig.tight_layout()
    plt.savefig("fov_integrated2_x.png", dpi=600, transparent=False)


if __name__ == "__main__":
    sc = sensitivity.SensitivityCalculator()
    time = Time('2024-04-20 12:00:00', scale="ut1")
    times = time + np.linspace(0, 24, 100) * u.hour

    for t in times:
        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(projection='mollweide'))
        draw_instantaneous_fov(ax, sc, t)
        draw_sources(ax)
        fig.tight_layout()

        plt.savefig(f"rnog_fov_{str(t)}.png", dpi=600, transparent=False)
        plt.close()