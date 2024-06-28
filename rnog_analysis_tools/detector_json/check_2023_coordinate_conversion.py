import numpy as np
import matplotlib.pyplot as plt
import glob
import rnog_analysis_tools.coordinate_system.coordinate_system

station_ids = [11, 12, 13, 21, 22, 23, 24]

folder_name = '/home/welling/Software/analysis-tools/rnog_analysis_tools/detector_json/gps_results/'
file_names = {
    "11": "20210810_station11_survey_results.csv",
    "12": "site_12_terrianiaq_survey_results_final.csv",
    "13": "site_13_ukaleq_survey_results.csv",
    "21": "20210810_station21_survey_results.csv",
    "22": "20210810_station22_survey_results.csv",
    "23": "20220714_station23_survey_results.csv",
    "24": "20220714_station24_latlon_survey_results.csv"
}
cs_disc = rnog_analysis_tools.coordinate_system.coordinate_system.CoordinateSystem('DISC', 2022)
cs_msf = rnog_analysis_tools.coordinate_system.coordinate_system.CoordinateSystem('GPS_basestation', 2022)
n_rows = 3
n_cols = len(station_ids) // n_rows
if len(station_ids) % n_rows > 0:
    n_cols += 1
fig1 = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
for i_station, station_id in enumerate(station_ids):
    ax1_1 = fig1.add_subplot(n_rows, n_cols, i_station + 1)
    ax1_1.set_aspect('equal')
    data_2022 = np.genfromtxt(
        folder_name + file_names[str(station_id)],
        delimiter=',',
        skip_header=1
    )
    station_positions_2022 = np.zeros((data_2022.shape[0], 3))
    for i_row in range(data_2022.shape[0]):
        station_positions_2022[i_row] = cs_disc.enu_to_enu(
            data_2022[i_row, 2],
            data_2022[i_row, 5],
            data_2022[i_row, 8],
            cs_msf.get_origin()
        )
    station_names_2022 = np.genfromtxt(
        folder_name + file_names[str(station_id)],
        delimiter=',',
        skip_header=1,
        dtype=str
    )

    station_positions_2023 = np.genfromtxt(
        '/home/welling/Software/analysis-tools/rnog_analysis_tools/detector_json/gps_results/survey_2023_station{}.csv'.format(
            station_id
        ),
        delimiter=',',
        skip_header=1
    )
    station_names_2023 = np.genfromtxt(
        '/home/welling/Software/analysis-tools/rnog_analysis_tools/detector_json/gps_results/survey_2023_station{}.csv'.format(
            station_id
        ),
        delimiter=',',
        skip_header=1,
        dtype=str
    )

    ax1_1.scatter(
        station_positions_2022[:, 0],
        station_positions_2022[:, 1],
        label='2022/21 survey',
        color='C0'
    )
    ax1_1.scatter(
        station_positions_2023[:, 2],
        station_positions_2023[:, 5],
        label='2023 survey',
        color='C1'
    )
    ax1_1.set_title('Station {}'.format(station_id))

    for i_name, name in enumerate(station_names_2022):
        ax1_1.text(
            station_positions_2022[i_name, 0] + 1,
            station_positions_2022[i_name, 1] + .5,
            name[0],
            color='C0'
        )
    for i_name, name in enumerate(station_names_2023):
        ax1_1.text(
            station_positions_2023[i_name, 2] + 1,
            station_positions_2023[i_name, 5] - .5,
            name[0],
            color='C1'
        )
    ax1_1.legend()

plt.show()

