import numpy as np
import matplotlib.pyplot as plt
import NuRadioReco.detector.detector
import astropy.time
import radiotools.helper



det = NuRadioReco.detector.detector.Detector(
    json_filename='RNO_season_2023.json'
)
det.update(astropy.time.Time.now())
station_ids = det.get_station_ids()

n_cols = 4
n_rows = len(station_ids) // n_cols
if len(station_ids) % n_cols > 0:
    n_rows += 1
fig1 = plt.figure(figsize=(n_cols * 3, n_rows * 3))
fig2 = plt.figure(figsize=(8, 8))
fig3 = plt.figure(figsize=(n_cols * 4, n_rows * 9))
ax2_1 = fig2.add_subplot(111)
global_channel_pos = []
station_positions = []
for i_station, station_id in enumerate(station_ids):
    ax1_1 = fig1.add_subplot(n_rows, n_cols, i_station +  1)
    ax1_1.set_title('Station {}'.format(station_id))
    ax3_1 = fig3.add_subplot(n_rows, n_cols, i_station + 1)
    ax3_1.set_title('Station {}'.format(station_id))

    deep_pos = []
    lpda_pos = []
    station_pos = det.get_absolute_position(station_id)
    for i_channel, channel_id in enumerate(det.get_channel_ids(station_id)):
        pos = det.get_relative_position(station_id, channel_id)
        antenna_angles = det.get_antenna_orientation(station_id, channel_id)
        antenna_orientation = radiotools.helper.spherical_to_cartesian(antenna_angles[0], antenna_angles[1])
        antenna_rotation = radiotools.helper.spherical_to_cartesian(antenna_angles[2], antenna_angles[3])
        if channel_id in range(12, 21):
            lpda_pos.append(pos)
            ax1_1.text(
                pos[0] + 1,
                pos[1],
                channel_id
            )
            ax1_1.plot(
                [pos[0], pos[0] + 3. * antenna_orientation[0]],
                [pos[1], pos[1] + 3. * antenna_orientation[1]],
                color='blue'
            )
            ax1_1.plot(
                [pos[0], pos[0] + 3. * antenna_rotation[0]],
                [pos[1], pos[1] + 3. * antenna_rotation[1]],
                color='red'
            )
        else:
            deep_pos.append(pos)
            if channel_id in [0, 9, 22]:
                ax1_1.text(
                    pos[0] + 1,
                    pos[1],
                    channel_id
                )
        global_channel_pos.append(pos + station_pos)
    solar1 = det.get_device(station_id, 51)
    ax1_1.scatter(solar1['ant_position_x'], solar1['ant_position_y'], c='k', marker='*')
    solar2 = det.get_device(station_id, 52)
    ax1_1.scatter(solar2['ant_position_x'], solar2['ant_position_y'], c='k', marker='*')

    surface_pulser = det.get_device(station_id, 2)
    fiber0 = det.get_device(station_id, 0)
    fiber1 = det.get_device(station_id, 1)
    ax1_1.scatter(
        surface_pulser['ant_position_x'],
        surface_pulser['ant_position_y'],
        c='orange'
    )
    ax3_1.scatter(
        surface_pulser['ant_position_x'],
        surface_pulser['ant_position_z'],
        c='orange',
        marker='s'
    )
    ax3_1.scatter(
        fiber0['ant_position_x'],
        fiber0['ant_position_z'],
        c='green',
        marker='s'
    )
    ax3_1.scatter(
        fiber1['ant_position_x'],
        fiber1['ant_position_z'],
        c='red',
        marker='s'
    )
    station_positions.append(station_pos)
    deep_pos = np.array(deep_pos)
    lpda_pos = np.array(lpda_pos)
    ax1_1.scatter(
        deep_pos[:, 0],
        deep_pos[:, 1]
    )
    if len(lpda_pos) > 0:
        ax1_1.scatter(
            lpda_pos[:, 0],
            lpda_pos[:, 1],
            marker='^'
        )
        ax3_1.scatter(
            lpda_pos[:, 0],
            lpda_pos[:, 2],
            marker='^'
        )
    ax1_1.set_aspect('equal')
    ax1_1.grid()
    ax3_1.scatter(
        deep_pos[:, 0],
        deep_pos[:, 2],
        s=10
    )
    ax3_1.grid()
station_positions = np.array(station_positions)
global_channel_pos = np.array(global_channel_pos)
for i_station, station_position in enumerate(station_positions):
    ax2_1.text(
        station_position[0] + 50,
        station_position[1],
        station_ids[i_station]
    )
ax2_1.scatter(
    global_channel_pos[:, 0],
    global_channel_pos[:, 1],
    s=1
)
ax2_1.set_aspect('equal')
ax2_1.grid()
fig1.tight_layout()
plt.show()