from multimessenger_tools import sensitivity

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt


sc = sensitivity.SensitivityCalculator()

# The BOAT - brightest of all time 
alert_time = Time('2020-10-09 13:16:59', scale="ut1")
boat_loc = SkyCoord.from_name('GRB221009A')

# to check whether source is in FOV:

in_fov = sc.is_in_fov(boat_loc, alert_time)
print(in_fov)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

sc.plot_sky_corr_in_altaz(axs[0], boat_loc, alert_time, [- 12 , 12] * u.h, time_scale="hr", title="")

sc.plot_effective_area(axs[1], boat_loc, alert_time, time_window=[- 1 , 2] * u.h, 
                        label=r"$RNO-G: ~ t \in t_0 [-1h, +2h], ~ all ~ flavors$", p_kwargs=dict(color="C3", ls="--", lw=5))

axs[1].legend()
fig.tight_layout()
plt.savefig("boat.png", transparent=True)
