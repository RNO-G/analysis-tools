import numpy as np
import json
import csv
import coordinate_system.coordinate_system
import radiotools.helper

"""
This script is used to build a JSON file containing an RNO-G detector descriptions to be used with NuRadioReco.
"""

"""
build_info.json contains the instructions on how to build the RNO-G JSON file
"""
build_instructions = json.load(open('build_info.json', 'r'))
"""
fiber_delays.json contains the measured delays for each fiber used for the deep channels.
It is created by the script fiber_delays.py, using the measurements done by Kaeli.
"""
fiber_delays = json.load(open('fiber_delays.json', 'r'))
channel_ids = np.arange(0, 24, dtype=int)
detector_description = {
    "channels": {},
    "stations": {}
}

cs_disk = coordinate_system.coordinate_system.CoordinateSystem()
cs_msf = coordinate_system.coordinate_system.CoordinateSystem('GPS_basestation')
i_channel = 0



for station_id in build_instructions.keys():
    if station_id == 'general':
        continue
    position_filename = build_instructions[station_id]['gps_results_file']
    position_reader = csv.reader(open(position_filename, 'r'), delimiter=',')
    power_string = None
    """
    Antenna coordinates are given relative to the power string. So we first find the power string in the GPS data
    """
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
    """
    Write the station information into the detector JSON
    """
    detector_description['stations'][str(station_id)] = {
        'station_id': int(station_id),
        'pos_easting': power_string_pos_disc[0],
        'pos_northing': power_string_pos_disc[1],
        'pos_altitude': power_string_pos_disc[2],
        'pos_site': 'summit',
        'commission_time': build_instructions[station_id]['deployment_dates']['Station'],
        'decommission_time': "{TinyDate}:2035-11-01T00:00:00"
    }
    """
    Loop through entries in the GPS file and write down the channels associated with them.
    """
    for row in position_reader:
        """
        The channel_associations lists which channels belong to a specific entry in the GPS data.
        """
        if row[0] in build_instructions['general']['channel_associations'].keys():
            if 'LPDA' in row[0]:
                amp_type = 'rno_surface'
                lpda_number = int(row[0].split(' ')[1])
                """
                The LPDAs are usually deployed together with the string on the same side of the station
                """
                if lpda_number < 4:
                    deployment_time = build_instructions[station_id]['deployment_dates']['Power A']
                elif lpda_number < 7:
                    deployment_time = build_instructions[station_id]['deployment_dates']['Helper C']
                else:
                    deployment_time = build_instructions[station_id]['deployment_dates']['Helper B']
            else:
                amp_type = 'iglu'
                deployment_time = build_instructions[station_id]['deployment_dates'][row[0]]
            """
            This transforms the GPS positions to be relative to the DISC hole instead of MSF
            """
            channel_position = np.array(cs_disk.enu_to_enu(
                float(row[2]),
                float(row[5]),
                float(row[8]),
                cs_msf.get_origin()
            ))
            relative_position = channel_position - power_string_pos_disc
            """
            Loop through all channels that are associated with the current GPS position
            """
            for channel_id in build_instructions['general']['channel_associations'][row[0]]:
                if channel_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]:
                    antenna_rotation_phi = 90.0
                    antenna_orientation_theta = 0.0
                    antenna_orientation_phi = 0.0
                    """
                    fiber_mapping specifies which box the fiber used for that station were taken from, so the right fiber
                    delay measurement is used
                    """
                    cable_delay = fiber_delays[str(build_instructions[station_id]['fiber_mapping'])][str(channel_id)]
                    """
                    Sometimes, fibers from a different box were used for one string. fiber_overrides lists the channels
                    for which a different fiber was used.
                    """
                    if str(channel_id) in build_instructions[station_id]['fiber_overrides'].keys():
                        cable_delay = fiber_delays[str(build_instructions[station_id]['fiber_overrides'][str(channel_id)])][str(channel_id)]
                else:
                    """
                    For the LPDAs, all cables essentially have the same fiber delays.
                    """
                    cable_delay = 45.5
                    """
                    Sometimes, the cables for the LPDAs were extended with jumper cables. If so, the channel ID is
                    stored in the jumpers entry once for every jumper cable added for that channel, so the additional
                    cable delays can be accounted for.
                    """
                    for jumper_channel in build_instructions[str(station_id)]['jumpers']:
                        if jumper_channel == channel_id:
                            cable_delay += 3.6
                    """
                    Take care of the way the LPDAs are oriented
                    """
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
                    """
                    We do not have depth measurements for the LPDAs, so we just take 0.5m for now.
                    """
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
                """
                Assign the correct antenna type to the channel
                """
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
