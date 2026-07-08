def get_station_config(station_id, default_station_config, station_specific_adjustments):
    cfg = default_station_config.copy()
    if str(station_id) in station_specific_adjustments:
        adjustments = station_specific_adjustments[str(station_id)]
        cfg.update(adjustments)
    return cfg
