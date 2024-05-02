import numpy as np
import json
import csv
import rnog_analysis_tools.coordinate_system.coordinate_system
import radiotools.helper
from NuRadioReco.utilities import units

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
surface_cable_delays = json.load(open('surface_cable_delays.json', 'r'))
channel_ids = np.arange(0, 24, dtype=int)
detector_description = {
    "channels": {},
    "stations": {}
}
year = 2023
cs_disk = rnog_analysis_tools.coordinate_system.coordinate_system.CoordinateSystem()
cs_msf = rnog_analysis_tools.coordinate_system.coordinate_system.CoordinateSystem('GPS_basestation')
i_channel = 0
i_devices = 0

def build_device(
        position,
        commission_time,
        station_id,
        device_id,
        comment
):
    dic = {
        'ant_comment': comment,
        'ant_orientation_phi': 0.0,
        'ant_orientation_theta': 0.0,
        'ant_rotation_phi': 90.0,
        'ant_rotation_theta': 90.0,
        'ant_position_x': position[0],
        'ant_position_y': position[1],
        'ant_position_z': position[2],
        'commission_time': commission_time,
        'decommission_time': '{TinyDate}:2035-11-01T00:00:00',
        'device_id': device_id,
        'station_id': station_id
    }
    return dic

surface_position = None
for station_id in build_instructions.keys():
    if station_id == 'general':
        continue
    if year == 2022:
        position_filename = build_instructions[station_id]['gps_results_file']
    else:
        position_filename = "gps_results/survey_2023_station{}.csv".format(station_id)
    position_reader = csv.reader(open(position_filename, 'r'), delimiter=',')
    power_string = None
    """
    Antenna coordinates are given relative to the power string. So we first find the power string in the GPS data
    """
    for row in csv.reader(open(position_filename, 'r'), delimiter=','):
        if row[0] == 'Power A':
            power_string = row
            break
    if year == 2022:
        power_string_pos_disc = np.array(cs_disk.enu_to_enu(
            float(power_string[2]),
            float(power_string[5]),
            float(power_string[8]),
            cs_msf.get_origin()
        ))
    else:
        power_string_pos_disc = np.array([
            float(power_string[2]),
            float(power_string[5]),
            float(power_string[8])
        ])
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
        Store position of the surface pulser
        """
        if row[0] == 'Surface Pulser':
            if year == 2022:
                surface_position = np.array(cs_disk.enu_to_enu(
                    float(row[2]),
                    float(row[5]),
                    float(row[8]),
                    cs_msf.get_origin()
                )) - power_string_pos_disc
            else:
                surface_position = np.array([
                    float(row[2]),
                    float(row[5]),
                    float(row[8])
                ]) - power_string_pos_disc


        if "Solar" in row[0]:
            print(row[0], i_devices)
            if year == 2022:
                pos = np.array(cs_disk.enu_to_enu(
                    float(row[2]),
                    float(row[5]),
                    float(row[8]),
                    cs_msf.get_origin()
                )) - power_string_pos_disc
            else:
                pos = np.array([
                    float(row[2]),
                    float(row[5]),
                    float(row[8])
                ]) - power_string_pos_disc
            if 'devices' not in detector_description.keys():
                detector_description['devices'] = {}
            detector_description['devices'][str(i_devices)] = {
                "station_id": int(station_id),
                "ant_comment": row[0],
                "ant_position_x": pos[0],
                "ant_position_y": pos[1],
                "ant_position_z": 2.,
                'pos_site': 'summit',
                'commission_time': build_instructions[station_id]['deployment_dates']['Station'],
                'decommission_time': "{TinyDate}:2035-11-01T00:00:00",
                "device_id": 50 + int(row[0][-1])

            }
            i_devices += 1
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
            if year == 2022:
                channel_position = np.array(cs_disk.enu_to_enu(
                    float(row[2]),
                    float(row[5]),
                    float(row[8]),
                    cs_msf.get_origin()
                ))
            else:
                channel_position = np.array([
                    float(row[2]),
                    float(row[5]),
                    float(row[8])
                ])
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
                    if channel_id in [0, 1, 2, 3]:
                        cable_delay += 1.768
                else:                
                    """
                    Get the cable delay measurements. The cables for the surface antennas were taken from the same box as the
                    fibers for the power string.
                    """
                    surface_cable_delay_measurement =surface_cable_delays[str(build_instructions[station_id]['fiber_mapping'])]
                    cable_delay = surface_cable_delay_measurement[str(build_instructions['general']['surface_cable_mappings'][str(channel_id)])]
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
                    if 'magnetic_correction' in build_instructions[station_id]:
                        if channel_id in build_instructions[station_id]['magnetic_correction']['plus']: 
                            antenna_rotation_phi += 23
                        if channel_id in build_instructions[station_id]['magnetic_correction']['minus']: 
                            antenna_rotation_phi -= 23
                    if channel_id in [19, 16, 13]:
                        antenna_orientation_theta = 0.0
                        antenna_orientation_phi = 0.0
                    else:
                        antenna_orientation_theta = 120.0
                        if int(station_id) == 21:
                            if channel_id in [14, 17, 20]:
                                antenna_orientation_phi = radiotools.helper.get_normalized_angle(
                                    antenna_rotation_phi + 90., True)
                            else:
                                antenna_orientation_phi = radiotools.helper.get_normalized_angle(
                                    antenna_rotation_phi - 90., True)
                        else:
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
    if 'devices' not in detector_description.keys():
        detector_description['devices'] = {}
    fiber_0_position = np.zeros(3)
    fiber_1_position = np.zeros(3)

    for i_ch, ch in detector_description['channels'].items():
        if ch['station_id'] == int(station_id) and ch['channel_id'] == 21:
            fiber_0_position[0] = ch['ant_position_x']
            fiber_0_position[1] = ch['ant_position_y']
            fiber_0_position[2] = -build_instructions[station_id]['channel_depths']['fiber0']
            detector_description['devices'][str(i_devices)] = build_device(
                fiber_0_position,
                ch['commission_time'],
                int(station_id),
                0,
                'Helper String C Cal Vpol'
            )
        if ch['station_id'] == int(station_id) and ch['channel_id'] == 11:
            fiber_1_position[0] = ch['ant_position_x']
            fiber_1_position[1] = ch['ant_position_y']
            fiber_1_position[2] = -build_instructions[station_id]['channel_depths']['fiber1']
            detector_description['devices'][str(i_devices + 1)] = build_device(
                fiber_1_position,
                ch['commission_time'],
                int(station_id),
                1,
                'Helper String B Cal Vpol'
            )
    surface_position[2] = -.5
    detector_description['devices'][str(i_devices + 2)] = build_device(
        surface_position,
        build_instructions[station_id]['deployment_dates']['Station'],
        int(station_id),
        2,
        'Surface Cal Pulser'
    )
    i_devices += 3
if year == 2022:
    outfile_name = 'RNO_season_2022.json'
else:
    outfile_name = 'RNO_season_2023.json'
json.dump(
    detector_description,
    open(outfile_name, 'w'),
    indent=2
)
