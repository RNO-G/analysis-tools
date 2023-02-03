import numpy as np
import csv
import coordinate_system.coordinate_system
site_names = {
    'site 1': 'station21',
    'site 2': 'station11',
    'site 3': 'station22'
}
cs_msf_2021 = coordinate_system.coordinate_system.CoordinateSystem('GPS_basestation', 2021)
cs_msf_2022 = coordinate_system.coordinate_system.CoordinateSystem('GPS_basestation', 2022)

srx_mapping = {
    '11': {
        '1': 9,
        '2': 8,
        '3': 7,
        '4': 6,
        '5': 5,
        '6': 4,
        '7': 3,
        '8': 2,
        '9': 1
    },
    '21': {
        '1': 4,
        '2': 5,
        '3': 6,
        '4': 7,
        '5': 8,
        '6': 9,
        '7': 1,
        '8': 2,
        '9': 3
    },
    '22': {
        '1': 4,
        '2': 5,
        '3': 6,
        '4': 7,
        '5': 8,
        '6': 9,
        '7': 1,
        '8': 2,
        '9': 3
    }
}

survey_data = np.genfromtxt(
    'gps_results/2021_survey_results.csv',
    delimiter=',',
    skip_header=1
)
survey_labels=np.genfromtxt(
'gps_results/2021_survey_results.csv',
    delimiter=',',
    skip_header=1,
    dtype=str
)
point_names = survey_labels[:, 0]
file_1 = open('gps_results/20210810_station11_survey_results.csv', 'w')
csv_1 = csv.writer(
    file_1,
    delimiter=','
)
csv_1.writerow(['Station Component', 'Filename', 'Easting Mean (m)', 'Std. Dev. Easting (m)', 'Precision Easting (m)',
                'Northing Mean (m)', 'Std. Dev. Northing (m)', 'Precision Northing (m)','Up Mean (m)', 'Std. Dev. Up (m)',
                'Precision Up (m)'
])
file_1.close()
file_2 = open('gps_results/20210810_station21_survey_results.csv', 'w')
csv_2 = csv.writer(
    file_2,
    delimiter=','
)
csv_2.writerow(['Station Component', 'Filename', 'Easting Mean (m)', 'Std. Dev. Easting (m)', 'Precision Easting (m)',
                'Northing Mean (m)', 'Std. Dev. Northing (m)', 'Precision Northing (m)','Up Mean (m)', 'Std. Dev. Up (m)',
                'Precision Up (m)'
])
file_2.close()
file_3 = open('gps_results/20210810_station22_survey_results.csv', 'w')
csv_3 = csv.writer(
    file_3,
    delimiter=','
)
csv_3.writerow(['Station Component', 'Filename', 'Easting Mean (m)', 'Std. Dev. Easting (m)', 'Precision Easting (m)',
                'Northing Mean (m)', 'Std. Dev. Northing (m)', 'Precision Northing (m)','Up Mean (m)', 'Std. Dev. Up (m)',
                'Precision Up (m)'
])
file_3.close()



for i_row, data_point in enumerate(survey_data):
    if 'site' in point_names[i_row]:
        station_name = site_names[point_names[i_row][0:6]]
        if 'srx' in point_names[i_row]:
            component_name = 'LPDA {}'.format(srx_mapping[station_name[-2:]][point_names[i_row][-1]])
        elif 'helper' in point_names[i_row]:
            component_name = 'Helper {}'.format(point_names[i_row][-1].upper())
        elif 'power' in point_names[i_row]:
            component_name = 'Power A'
        elif point_names[i_row][-6:] == 'pulser':
            component_name = 'Surface Pulser'
        else:
            continue
        coordinates_2021 = survey_data[i_row, 1:4]
        coordinates_2022 = np.array(cs_msf_2022.enu_to_enu(
            coordinates_2021[0],
            coordinates_2021[1],
            coordinates_2021[2],
            cs_msf_2021.get_origin()
        ))
        with open('gps_results/20210810_{}_survey_results.csv'.format(station_name), 'a') as csv_file:
            csv_writer = csv.writer(
                csv_file,
                delimiter=','
            )
            csv_writer.writerow([
                component_name,
                '2021_survey_results.csv',
                coordinates_2022[0],
                'nan',
                survey_data[i_row, 5],
                coordinates_2022[1],
                'nan',
                survey_data[i_row, 5],
                coordinates_2022[2],
                'nan',
                survey_data[i_row, 6]

            ])
            csv_file.close()
