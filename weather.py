import os
import numpy as np
import xarray as xr
from geopy.distance import geodesic

def nearest_index(array, value):
    array = np.array(array)
    return (np.abs(array - value)).argmin()

def pressure_to_feet(p_hpa):
    p0 = 1013.25
    meters = 44330 * (1 - (p_hpa / p0) ** 0.1903)
    feet = meters * 3.28084
    return feet

def get_grey_shade(percent):
    if percent < 10:
        return None
    shade = int(250 - ((percent - 10) / 90) * 10)
    return max(240, min(250, shade))

def colored_block(percent):
    shade = get_grey_shade(percent)
    if shade is None:
        return " "
    return f"\033[38;5;{shade}m█\033[0m"

def interpolate_lat_lon(start_lat, start_lon, end_lat, end_lon, num_points):
    lats = np.linspace(start_lat, end_lat, num_points)
    lons = np.linspace(start_lon, end_lon, num_points)
    return list(zip(lats, lons))

def deduplicate_by_grid(points, lat_vals, lon_vals):
    seen = set()
    unique = []
    for lat, lon in points:
        lon_360 = lon % 360
        lat_idx = nearest_index(lat_vals, lat)
        lon_idx = nearest_index(lon_vals, lon_360)
        key = (lat_idx, lon_idx)
        if key not in seen:
            seen.add(key)
            unique.append((lat, lon, lat_idx, lon_idx))
    return unique

def resample_columns_to_width(columns, target_width):
    if len(columns) == 0:
        return [[" "] * target_width] * len(columns[0])  # fallback
    if len(columns) == target_width:
        return columns
    indices = np.linspace(0, len(columns) - 1, target_width, dtype=int)
    return [columns[i] for i in indices]

def main(start_lat, start_lon, end_lat, end_lon):
    grib_file = "gfs.t00z.pgrb2.0p25.f000"

    if not os.path.exists(grib_file):
        print(f"Error: File {grib_file} not found.")
        return

    print(f"Opening {grib_file}...")

    ds = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        filter_by_keys={'shortName': 'tcc', 'typeOfLevel': 'isobaricInhPa'},
        backend_kwargs={'indexpath': ''},
        decode_timedelta=True
    )

    lat_vals = ds['latitude'].values
    lon_vals = ds['longitude'].values
    pressures = ds['isobaricInhPa'].values

    # Step 1: Interpolate many points along route
    interpolated = interpolate_lat_lon(start_lat, start_lon, end_lat, end_lon, 300)

    # Step 2: Deduplicate by GFS grid cell
    unique_points = deduplicate_by_grid(interpolated, lat_vals, lon_vals)
    print(f"Unique grid points found: {len(unique_points)}")

    # Step 3: Query cloud coverage for each unique point
    tcc_columns = []
    for _, _, lat_idx, lon_idx in unique_points:
        tcc_slice = ds['tcc'].isel(latitude=lat_idx, longitude=lon_idx)
        tcc_columns.append(tcc_slice.values)

    # Step 4: Resample to exactly 100 columns
    tcc_columns = resample_columns_to_width(tcc_columns, 100)

    # Step 5: Transpose to rows (one per pressure level)
    tcc_by_altitude = list(zip(*tcc_columns))

    print("\nCloud Coverage Graph (x = location, y = altitude):")
    for p, row in reversed(list(zip(pressures, tcc_by_altitude))):
        altitude_ft = pressure_to_feet(p)
        bars = "".join(colored_block(val) for val in row)
        print(f"{p:4.0f} hPa (~{altitude_ft:5.0f} ft): {bars}")

if __name__ == "__main__":
    main(42.452858, -75.063774, 41.334782, -77.360023)
