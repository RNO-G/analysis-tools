import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import rnog_analysis_tools.coordinate_system.coordinate_system


station_ids = [11, 12, 13, 21, 22, 23, 24]
file_path = '/home/welling/RadioNeutrino/data/GPS_SURVEY_DATA_2023/'
name_lists = json.load(open('survey_names_2023.json', 'r'))
cs = rnog_analysis_tools.coordinate_system.coordinate_system.CoordinateSystem(year=2023)

for station_id in station_ids:
    output_writer = csv.writer(open('gps_results/survey_2023_station{}.csv'.format(station_id), 'w'))
    file_name = file_path + 'station{}/KaeliPhone_Station{}.csv'.format(station_id, station_id)
    data = np.genfromtxt(
        file_name,
        delimiter=',',
        skip_header=1
    )
    position_names = np.genfromtxt(
        file_name,
        delimiter=',',
        skip_header=1,
        dtype=str
    )[:, 1]
    output_writer.writerow([
        'Station Component',
        'Filename,Easting Mean (m)',
        'Std. Dev. Easting (m)',
        'Precision Easting (m)',
        'Northing Mean (m)',
        'Std. Dev. Northing (m)',
        'Precision Northing (m)',
        'Up Mean (m)',
        'Std. Dev. Up (m)',
        'Precision Up (m)'
    ])
    for name in name_lists[str(station_id)]:
        mean_pos = np.zeros(3)
        n_points = 0
        for i_pos, pos in enumerate(position_names):
            if pos in name_lists[str(station_id)][name]:
                mean_pos += cs.geodetic_to_enu(*data[i_pos, 4:7])
                n_points += 1
        if n_points > 0:
            mean_pos /= n_points
        output_writer.writerow([
            name,
            '',
            mean_pos[0],
            'nan',
            'nan',
            mean_pos[1],
            'nan',
            'nan',
            mean_pos[2],
            'nan',
            'nan'
        ])