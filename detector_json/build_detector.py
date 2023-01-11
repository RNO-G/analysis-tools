import numpy as np
import json
import csv
import coordinate_system.coordinate_system
import radiotools.helper

build_instructions = json.load(open('build_info.json', 'r'))
fiber_delays = json.load(open('fiber_delays.json', 'r'))
channel_ids = np.arange(0, 24, dtype=int)
detector_description = {
    "channels": {},
    "stations": {}
}
nominal_station_positions = np.genfromtxt(
    'gps_results/labeled-stations.csv',
    delimiter=',',
    skip_header=1
)
cs_disk = coordinate_system.coordinate_system.CoordinateSystem()
cs_msf = coordinate_system.coordinate_system.CoordinateSystem('GPS_basestation')
i_channel = 0
i_station = 0

def get_nominal_station_position(station_id):
    for row in nominal_station_positions:
        if int(row[0]) == station_id:
            return np.array(cs_disk.geodetic_to_enu(row[4], row[3]))
    return None


for station_id in build_instructions.keys():
    if station_id == 'general':
        continue
    position_filename = build_instructions[station_id]['gps_results_file']
    position_reader = csv.reader(open(position_filename, 'r'), delimiter=',')
    power_string = None
    for row in csv.reader(open(position_filename, 'r'), delimiter=','):
        if row[0] == 'Power A':
            power_string = row
            break
    power_string_pos_disc = np.array(cs_disk.enu_to_enu(
        float(power_string[2]),
        float(power_string[5]),
        float(power_string[8]),
        cs_msf.get_origin()
    ))
    detector_description['stations'][str(station_id)] = {
        'station_id': int(station_id),
        'pos_easting': power_string_pos_disc[0],
        'pos_northing': power_string_pos_disc[1],
        'pos_altitude': power_string_pos_disc[2],
        'pos_site': 'summit',
        'commission_time': build_instructions[station_id]['deployment_dates']['Station'],
        'decommission_time': "{TinyDate}:2035-11-01T00:00:00"
    }
    for row in position_reader:
        if row[0] in build_instructions['general']['channel_associations'].keys():
            if 'LPDA' in row[0]:
                amp_type = 'rno_surface'
                lpda_number = int(row[0].split(' ')[1])
                if lpda_number < 4:
                    deployment_time = build_instructions[station_id]['deployment_dates']['Power A']
                elif lpda_number < 7:
                    deployment_time = build_instructions[station_id]['deployment_dates']['Helper B']
                else:
                    deployment_time = build_instructions[station_id]['deployment_dates']['Helper C']
            else:

                amp_type = 'iglu'
                deployment_time = build_instructions[station_id]['deployment_dates'][row[0]]
            channel_position = np.array(cs_disk.enu_to_enu(
                float(row[2]),
                float(row[5]),
                float(row[8]),
                cs_msf.get_origin()
            ))
            relative_position = channel_position - power_string_pos_disc
            for channel_id in build_instructions['general']['channel_associations'][row[0]]:
                if channel_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
                    antenna_rotation_phi = 90.0
                    antenna_orientation_theta = 0.0
                    antenna_orientation_phi = 0.0
                    cable_delay = fiber_delays[str(build_instructions[station_id]['fiber_mapping'])][str(channel_id)]
                    if str(channel_id) in build_instructions[station_id]['fiber_overrides'].keys():
                        cable_delay = fiber_delays[str(build_instructions[station_id]['fiber_overrides'][str(channel_id)])][str(channel_id)]
                else:
                    cable_delay = 45.5
                    for jumper_channel in build_instructions[str(station_id)]['jumpers']:
                        if jumper_channel == channel_id:
                            cable_delay += 3.6
                    antenna_rotation_phi = build_instructions[station_id]['lpda_rotations'][str(channel_id)]
                    if channel_id in [19, 16, 13]:
                        antenna_orientation_theta = 0.0
                        antenna_orientation_phi = 0.0
                    else:
                        antenna_orientation_theta = 120.0
                        if channel_id in [14, 17, 20]:
                            antenna_orientation_phi = radiotools.helper.get_normalized_angle(antenna_rotation_phi - 90., True)
                        else:
                            antenna_orientation_phi = radiotools.helper.get_normalized_angle(antenna_rotation_phi + 90., True)
                if str(channel_id) in build_instructions[str(station_id)]['channel_depths'].keys():
                    channel_z = -build_instructions[str(station_id)]['channel_depths'][str(channel_id)]
                else:
                    channel_z = -.5

                channel_json = {
                    'station_id': int(station_id),
                    'channel_id': channel_id,
                    "ant_rotation_phi": antenna_rotation_phi,
                    "ant_rotation_theta": 90.0,
                    "ant_orientation_phi": antenna_orientation_phi,
                    "ant_orientation_theta": antenna_orientation_theta,
                    "ant_position_x": relative_position[0],
                    "ant_position_y": relative_position[1],
                    "ant_position_z": channel_z,
                    'amp_type': amp_type,
                    'cab_time_delay': cable_delay,
                    'adc_n_samples': 2048,
                    'adc_sampling_frequency': 3.2,
                    'commission_time': deployment_time,
                    'decommission_time': "{TinyDate}:2035-11-01T00:00:00"
                }
                for antenna_type in build_instructions['general']['antenna_types'].keys():
                    if channel_id in build_instructions['general']['antenna_types'][antenna_type]:
                        channel_json['ant_type'] = antenna_type

                detector_description['channels'][str(i_channel)] = channel_json
                i_channel += 1
json.dump(
    detector_description,
    open('RNO_season_2022.json', 'w'),
    indent=2
)
