#!/usr/bin/env python3
"""
P2P Energy Trading Dataset Generator with PVLib
Generates synthetic load and real solar generation data for Indian households

With configurable prosumer/consumer/balanced ratios for realistic P2P trading

Requirements:
    pip install pandas numpy pvlib

Usage:
    python generate_dataset.py

Or customize:
    from generate_dataset import generate_p2p_dataset

    summary = generate_p2p_dataset(
        n_households=50,
        n_days=30,
        interval_minutes=15,
        prosumer_ratio=0.3,      # 30% prosumers (exporters)
        consumer_ratio=0.4,      # 40% consumers (importers)
        balanced_ratio=0.3,      # 30% balanced
        output_directory='my_data'
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

def create_indian_load_profile_template():
    """Create normalized 24-hour load profile for Indian residential patterns"""
    hourly_profile = np.array([
        0.32, 0.28, 0.26, 0.25, 0.26, 0.30,  # 00:00-05:00 Night
        0.42, 0.58, 0.65, 0.55, 0.48, 0.45,  # 06:00-11:00 Morning peak
        0.44, 0.46, 0.43, 0.45, 0.50, 0.58,  # 12:00-17:00 Afternoon
        0.72, 0.85, 0.95, 1.00, 0.82, 0.55   # 18:00-23:00 Evening peak
    ])
    return hourly_profile / hourly_profile.sum()

def create_weekday_weekend_variants(base_profile):
    """Create weekday and weekend load variants"""
    weekday = base_profile.copy()
    weekend = base_profile.copy()
    weekend[10:17] *= 1.15
    weekend[7:9] *= 0.9
    weekend[9:11] *= 1.1
    weekend = weekend / weekend.sum()
    return weekday, weekend

def add_power_outages(load_series, timestamps, outage_probability=0.15):
    """Add random power outages (15% probability per day, 1-6 hours duration)"""
    load_modified = load_series.copy()
    outage_flags = np.zeros(len(load_series), dtype=bool)

    df_temp = pd.DataFrame({
        'timestamp': timestamps, 
        'index': np.arange(len(timestamps))
    })
    df_temp['date'] = df_temp['timestamp'].dt.date

    for date in df_temp['date'].unique():
        if np.random.random() < outage_probability:
            duration_hours = np.random.choice([1, 2, 3, 4, 6], 
                                             p=[0.3, 0.3, 0.2, 0.15, 0.05])
            possible_starts = list(range(6, 24)) + [0]
            start_hour = np.random.choice(possible_starts)

            day_indices = df_temp[df_temp['date'] == date]['index'].values
            day_hours = df_temp[df_temp['date'] == date]['timestamp'].dt.hour.values

            for h in range(duration_hours):
                target_hour = (start_hour + h) % 24
                hour_mask = day_hours == target_hour
                outage_indices = day_indices[hour_mask]

                if len(outage_indices) > 0:
                    load_modified[outage_indices] = 0
                    outage_flags[outage_indices] = True

    return load_modified, outage_flags

def generate_household_load(household_id, num_days, interval_minutes, 
                           monthly_kwh, weekday_profile, weekend_profile):
    """Generate load profile for one household"""
    daily_kwh = monthly_kwh / 30.0
    start_date = datetime(2024, 6, 1, 0, 0, 0)

    timestamps = pd.date_range(
        start=start_date,
        periods=num_days * 24 * (60 // interval_minutes),
        freq=f'{interval_minutes}min'
    )

    load_kw = np.zeros(len(timestamps))

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5

        daily_profile = weekend_profile if is_weekend else weekday_profile
        daily_load = daily_profile * daily_kwh
        daily_multiplier = np.random.uniform(0.8, 1.2)
        daily_load = daily_load * daily_multiplier

        intervals_per_day = 24 * (60 // interval_minutes)

        if interval_minutes == 60:
            load_interpolated = daily_load
        else:
            hourly_indices = np.arange(24)
            interval_indices = np.linspace(0, 23, intervals_per_day)
            load_interpolated = np.interp(interval_indices, hourly_indices, daily_load)

        start_idx = day * intervals_per_day
        end_idx = start_idx + intervals_per_day
        load_kw[start_idx:end_idx] = load_interpolated

    return timestamps, load_kw

def generate_solar_with_pvlib(latitude, longitude, num_days, interval_minutes, 
                               pv_capacity_kw, start_date='2024-06-01'):
    """Generate solar using PVGIS data (2005-2020)"""
    try:
        import pvlib

        pvgis_year = 2020

        result = pvlib.iotools.get_pvgis_hourly(
            latitude=latitude,
            longitude=longitude,
            start=pvgis_year,
            end=pvgis_year,
            pvcalculation=True,
            peakpower=pv_capacity_kw,
            surface_tilt=20,
            surface_azimuth=180,
            loss=14,
            url='https://re.jrc.ec.europa.eu/api/v5_2/'
        )

        if len(result) == 3:
            data, inputs, meta = result
        else:
            data, meta = result

        start_timestamp = pd.Timestamp(f'{pvgis_year}-06-01', tz='UTC')
        end_timestamp = start_timestamp + pd.Timedelta(days=num_days)

        mask = (data.index >= start_timestamp) & (data.index < end_timestamp)
        data_filtered = data.loc[mask]

        solar_power_w = data_filtered['P'].values
        solar_kw_hourly = solar_power_w / 1000.0

        if interval_minutes == 60:
            target_length = num_days * 24
            solar_kw = solar_kw_hourly[:target_length]

        elif interval_minutes in [15, 30]:
            intervals_per_hour = 60 // interval_minutes
            hourly_timestamps = np.arange(len(solar_kw_hourly))
            subhourly_timestamps = np.linspace(0, len(solar_kw_hourly) - 1, 
                                               len(solar_kw_hourly) * intervals_per_hour)

            solar_kw_interpolated = np.interp(subhourly_timestamps, 
                                              hourly_timestamps, 
                                              solar_kw_hourly)

            target_length = num_days * 24 * intervals_per_hour
            solar_kw = solar_kw_interpolated[:target_length]
        else:
            raise ValueError(f"interval_minutes must be 15, 30, or 60")

        return solar_kw, True

    except ImportError:
        print("  ⚠ pvlib not installed: pip install pvlib")
        return None, False
    except Exception as e:
        error_str = str(e)
        print(f"  ⚠ pvlib error: {error_str[:80]}")
        return None, False

def generate_synthetic_solar_fallback(num_days, interval_minutes, pv_capacity_kw):
    """Fallback synthetic solar profile"""
    intervals_per_day = 24 * (60 // interval_minutes)
    total_intervals = num_days * intervals_per_day
    solar_kw = np.zeros(total_intervals)

    sunrise_hour, sunset_hour = 5.5, 19.2

    for day in range(num_days):
        daily_factor = np.random.uniform(0.85, 1.0)

        for interval in range(intervals_per_day):
            hour = (interval * interval_minutes) / 60.0

            if sunrise_hour <= hour <= sunset_hour:
                day_fraction = (hour - sunrise_hour) / (sunset_hour - sunrise_hour)
                solar_fraction = np.sin(np.pi * day_fraction) ** 1.5
                horizon_factor = 1.0 - 0.3 * np.exp(-5 * (day_fraction - 0.5)**2)
                temp_profile = 25 + 17 * np.sin(np.pi * day_fraction)
                temp_derating = 1.0 - 0.004 * max(0, temp_profile - 25)

                solar_output = (pv_capacity_kw * solar_fraction * 
                               horizon_factor * temp_derating * daily_factor)

                solar_kw[day * intervals_per_day + interval] = max(0, solar_output)

    return solar_kw, False

def generate_solar_generation(latitude, longitude, num_days, interval_minutes, 
                              pv_capacity_kw, start_date='2024-06-01'):
    """Main solar generation function"""
    solar_kw, success = generate_solar_with_pvlib(
        latitude, longitude, num_days, interval_minutes, 
        pv_capacity_kw, start_date
    )

    if success:
        print(f"  ✓ Real solar data from PVGIS (pvlib)")
        return solar_kw
    else:
        print(f"  ⚠ Using synthetic solar profile")
        solar_kw, _ = generate_synthetic_solar_fallback(
            num_days, interval_minutes, pv_capacity_kw
        )
        return solar_kw

def save_household_data(output_dir, household_id, timestamps, load_kw, solar_kw, latitude, longitude):
    """Save household data to CSV, with PVLib coordinates and time_of_day."""
    import pandas as pd
    from pathlib import Path

    dt_index = pd.DatetimeIndex(timestamps)
    df = pd.DataFrame({
        'timestamp': timestamps,
        'household_id': household_id,
        'load_kW': load_kw,
        'solar_kW': solar_kw,
        'latitude': latitude,  # constant for all rows per household
        'longitude': longitude,  # constant for all rows per household
        'time_of_day': dt_index.hour + dt_index.minute / 60  # decimal hour
    })
    filepath = Path(output_dir) / f'{household_id}.csv'
    df.to_csv(filepath, index=False, float_format='%.4f')
    return filepath


def generate_realistic_household_mix(n_households, prosumer_ratio=0.33, 
                                     consumer_ratio=0.33, balanced_ratio=0.34):
    """
    Create realistic household mix for P2P trading

    Parameters:
    -----------
    n_households : int
        Total number of households
    prosumer_ratio : float
        Fraction of prosumers (0-1), exporters
        Low consumption (80-120 kWh/month), large PV (2.5-3.5 kW)
    consumer_ratio : float
        Fraction of consumers (0-1), importers
        High consumption (200-280 kWh/month), small PV (0.5-1.2 kW)
    balanced_ratio : float
        Fraction of balanced households (0-1)
        Medium consumption (140-180 kWh/month), medium PV (1.5-2.5 kW)

    Returns:
    --------
    households : list of dicts with 'id', 'monthly_kwh', 'pv_kw', 'type'
    """

    # Validate ratios
    total_ratio = prosumer_ratio + consumer_ratio + balanced_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Calculate number of households in each category
    n_prosumers = round(n_households * prosumer_ratio)
    n_consumers = round(n_households * consumer_ratio)
    n_balanced = n_households - n_prosumers - n_consumers

    households = []

    # PROSUMERS: Low consumption, high PV (net exporters)
    for i in range(n_prosumers):
        hh_id = f"HH{len(households)+1:03d}"
        monthly_kwh = np.random.uniform(80, 120)      # Low consumption
        pv_kw = np.random.uniform(2.5, 3.5)           # Large PV
        households.append({
            'id': hh_id, 
            'monthly_kwh': monthly_kwh, 
            'pv_kw': pv_kw,
            'type': 'prosumer'
        })

    # CONSUMERS: High consumption, small PV (net importers)
    for i in range(n_consumers):
        hh_id = f"HH{len(households)+1:03d}"
        monthly_kwh = np.random.uniform(200, 280)     # High consumption
        pv_kw = np.random.uniform(0.5, 1.2)           # Small PV
        households.append({
            'id': hh_id, 
            'monthly_kwh': monthly_kwh, 
            'pv_kw': pv_kw,
            'type': 'consumer'
        })

    # BALANCED: Medium consumption, medium PV (flexible)
    for i in range(n_balanced):
        hh_id = f"HH{len(households)+1:03d}"
        monthly_kwh = np.random.uniform(140, 180)     # Medium consumption
        pv_kw = np.random.uniform(1.5, 2.5)           # Medium PV
        households.append({
            'id': hh_id, 
            'monthly_kwh': monthly_kwh, 
            'pv_kw': pv_kw,
            'type': 'balanced'
        })

    return households

def generate_p2p_dataset(n_households, n_days, interval_minutes, 
                         output_directory='example_dataset',
                         location='delhi',
                         prosumer_ratio=0.33,
                         consumer_ratio=0.33,
                         balanced_ratio=0.34,
                         include_outages=True, 
                         random_seed=42):
    """
    Main function to generate complete P2P dataset

    Parameters:
    -----------
    n_households : int
        Number of households (e.g., 10, 50, 100)
    n_days : int
        Simulation duration in days (e.g., 7, 30, 365)
    interval_minutes : int
        Time resolution: 15, 30, or 60 minutes
    output_directory : str
        Output folder name
    location : str or tuple
        Either 'delhi' or (latitude, longitude) tuple
    prosumer_ratio : float (0-1)
        Fraction of prosumers (exporters)
    consumer_ratio : float (0-1)
        Fraction of consumers (importers)
    balanced_ratio : float (0-1)
        Fraction of balanced households
    include_outages : bool
        Include random power outages
    random_seed : int
        For reproducibility

    Returns:
    --------
    summary : dict with generation statistics
    """

    np.random.seed(random_seed)
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    if location == 'delhi':
        latitude, longitude = 28.6, 77.2
        loc_name = "Delhi"
    else:
        latitude, longitude = location
        loc_name = f"({latitude:.2f}°N, {longitude:.2f}°E)"

    print(f"\n{'='*70}")
    print(f"P2P Energy Trading Data Generator with PVLib")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Households:        {n_households}")
    print(f"  Duration:          {n_days} days")
    print(f"  Time interval:     {interval_minutes} minutes")
    print(f"  Location:          {loc_name}")
    print(f"  Prosumer ratio:    {prosumer_ratio*100:.0f}%")
    print(f"  Consumer ratio:    {consumer_ratio*100:.0f}%")
    print(f"  Balanced ratio:    {balanced_ratio*100:.0f}%")
    print(f"  Power outages:     {include_outages}")
    print(f"  Output directory:  {output_dir.absolute()}")
    print(f"{'='*70}\n")

    # Create load profiles
    print("Creating Indian residential load profiles...")
    base_profile = create_indian_load_profile_template()
    weekday_profile, weekend_profile = create_weekday_weekend_variants(base_profile)
    print(f"  ✓ Morning peak: 7-9 AM, Evening peak: 8-11 PM\n")

    # Generate realistic household mix
    print("Creating realistic P2P household mix...")
    households = generate_realistic_household_mix(
        n_households, prosumer_ratio, consumer_ratio, balanced_ratio
    )

    prosumers = sum(1 for h in households if h['type'] == 'prosumer')
    consumers = sum(1 for h in households if h['type'] == 'consumer')
    balanced = sum(1 for h in households if h['type'] == 'balanced')

    print(f"  Prosumers (exporters): {prosumers} households")
    print(f"  Consumers (importers): {consumers} households")
    print(f"  Balanced (flexible):   {balanced} households\n")

    summary = {
        'configuration': {
            'n_households': n_households,
            'n_days': n_days,
            'interval_minutes': interval_minutes,
            'location': loc_name,
            'prosumer_ratio': prosumer_ratio,
            'consumer_ratio': consumer_ratio,
            'balanced_ratio': balanced_ratio,
            'include_outages': include_outages
        },
        'households': [],
        'total_files': 0,
        'total_records': 0,
        'net_summary': {
            'exporters': 0,
            'importers': 0,
            'total_surplus': 0.0,
            'total_deficit': 0.0
        }
    }

    print(f"Generating data for {n_households} households...\n")

    for idx, hh in enumerate(households):
        print(f"[{idx+1:3d}/{n_households}] {hh['id']} ({hh['type']})")
        print(f"  Monthly target:  {hh['monthly_kwh']:6.1f} kWh")
        print(f"  PV capacity:     {hh['pv_kw']:6.2f} kW")

        timestamps, load_kw = generate_household_load(
            hh['id'], n_days, interval_minutes, 
            hh['monthly_kwh'], weekday_profile, weekend_profile
        )

        if include_outages:
            load_kw, outage_flags = add_power_outages(load_kw, timestamps, 0.15)
            num_outages = len(np.where(np.diff(outage_flags.astype(int)) == 1)[0])
            print(f"  Outages added:   {num_outages} events")
        else:
            num_outages = 0

        solar_kw = generate_solar_generation(
            latitude, longitude, n_days, interval_minutes, 
            hh['pv_kw'], start_date='2024-06-01'
        )

        min_len = min(len(load_kw), len(solar_kw))
        load_kw = load_kw[:min_len]
        solar_kw = solar_kw[:min_len]
        timestamps = timestamps[:min_len]

        filepath = save_household_data(output_dir, hh['id'], timestamps, load_kw, solar_kw, latitude, longitude)

        energy_factor = interval_minutes / 60
        total_load_kwh = np.sum(load_kw) * energy_factor
        total_solar_kwh = np.sum(solar_kw) * energy_factor
        net_kwh = total_solar_kwh - total_load_kwh

        # Track net energy direction
        if net_kwh > 0:
            summary['net_summary']['exporters'] += 1
            summary['net_summary']['total_surplus'] += net_kwh
        else:
            summary['net_summary']['importers'] += 1
            summary['net_summary']['total_deficit'] += abs(net_kwh)

        print(f"  Total load:      {total_load_kwh:6.1f} kWh")
        print(f"  Total solar:     {total_solar_kwh:6.1f} kWh")
        print(f"  Net:             {net_kwh:+6.1f} kWh {'(export)' if net_kwh > 0 else '(import)'}")
        print(f"  Saved to:        {filepath.name}\n")

        summary['households'].append({
            'household_id': hh['id'],
            'type': hh['type'],
            'monthly_kwh_target': round(hh['monthly_kwh'], 2),
            'pv_kw': round(hh['pv_kw'], 2),
            'total_load_kwh': round(total_load_kwh, 2),
            'total_solar_kwh': round(total_solar_kwh, 2),
            'net_kwh': round(net_kwh, 2),
            'num_outages': num_outages,
            'records': len(timestamps)
        })

        summary['total_records'] += len(timestamps)
        summary['total_files'] += 1

    # Save summary
    summary_file = output_dir / 'dataset_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"{'='*70}")
    print(f"✓ Dataset Generation Complete!")
    print(f"{'='*70}")
    print(f"Output Summary:")
    print(f"  Total files:           {summary['total_files']}")
    print(f"  Total records:         {summary['total_records']:,}")
    print(f"  Records per household: {summary['total_records'] // n_households:,}")
    print(f"")
    print(f"P2P Market Structure:")
    print(f"  Exporters (surplus):   {summary['net_summary']['exporters']}")
    print(f"  Importers (deficit):   {summary['net_summary']['importers']}")
    print(f"  Total export capacity: {summary['net_summary']['total_surplus']:,.1f} kWh")
    print(f"  Total import need:     {summary['net_summary']['total_deficit']:,.1f} kWh")
    print(f"  Trading potential:     {min(summary['net_summary']['total_surplus'], summary['net_summary']['total_deficit']):,.1f} kWh")
    print(f"")
    print(f"Files:")
    print(f"  Location: {output_dir.absolute()}")
    print(f"  Summary:  {summary_file.name}")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    # Example 1: 33-33-34 split (default)
    summary = generate_p2p_dataset(
        n_households=10,
        n_days=7,
        interval_minutes=60, #Must be 15, 30 or 60
        prosumer_ratio=0.33,
        consumer_ratio=0.33,
        balanced_ratio=0.34
    )

    # Example 2: More exporters (50-25-25 split)
    # summary = generate_p2p_dataset(
    #     n_households=20,
    #     n_days=30,
    #     interval_minutes=15,
    #     prosumer_ratio=0.50,
    #     consumer_ratio=0.25,
    #     balanced_ratio=0.25
    # )

    # Example 3: More importers (20-60-20 split)
    # summary = generate_p2p_dataset(
    #     n_households=50,
    #     n_days=30,
    #     interval_minutes=60,
    #     prosumer_ratio=0.20,
    #     consumer_ratio=0.60,
    #     balanced_ratio=0.20
    # )

    print("Dataset ready for P2P trading simulation!")
