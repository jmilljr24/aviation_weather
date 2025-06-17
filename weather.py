import os
import numpy as np
import xarray as xr
import requests
import matplotlib.pyplot as plt
from math import radians, cos, sin, sqrt, atan2
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import requests
import datetime
from pathlib import Path

def get_latest_available_gfs_file(forecast_hour):
    now = datetime.datetime.utcnow()
    cycles = ["18", "12", "06", "00"]
    max_days_back = 2

    for day_offset in range(max_days_back + 1):
        date = now - datetime.timedelta(days=day_offset)
        date_str = date.strftime("%Y%m%d")

        for cycle in cycles:
            file_name = f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour}"
            url = (
                f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
                f"gfs.{date_str}/{cycle}/atmos/{file_name}"
            )

            print(f"Trying: {url}")
            response = requests.head(url)
            if response.status_code == 200:
                print(f"Found available file: {file_name}")
                return download_gfs_file(date_str, cycle, forecast_hour)

    raise RuntimeError("No available GFS data found in the last few days.")

def download_gfs_file(date_str, cycle, forecast_hour):
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{date_str}/{cycle}/atmos"
    file_name = f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour}"
    local_path = Path(file_name)

    if local_path.exists():
        print(f"{file_name} already exists.")
        return file_name

    print(f"Downloading {file_name} from {base_url}...")
    url = f"{base_url}/{file_name}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download GFS file: {url} (HTTP {response.status_code})")

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded {file_name}")
    return file_name

def get_airport_latlon(code):
    code = code.upper().strip()
    if len(code) == 3:
        query_field = "IDENT"
    elif len(code) == 4:
        query_field = "ICAO_ID"
    else:
        raise ValueError(f"Invalid airport code: {code}")

    url = (
        "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/US_Airport/FeatureServer/0/query"
        f"?where={query_field}+%3D+%27{code}%27"
        "&outFields=IDENT,NAME,LATITUDE,LONGITUDE,ICAO_ID"
        "&outSR=4326"
        "&f=json"
    )

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data for {code}: HTTP {response.status_code}")

    data = response.json()
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No location data found for airport code: {code}")

    geometry = features[0]["geometry"]
    lat = geometry["y"]
    lon = geometry["x"]
    return lat, lon
def nearest_index(array, value):
    array = np.array(array)
    return (np.abs(array - value)).argmin()

def pressure_to_feet(p_hpa):
    p0 = 1013.25
    meters = 44330 * (1 - (p_hpa / p0) ** 0.1903)
    feet = meters * 3.28084
    return feet

def feet_to_pressure(feet):
    meters = feet / 3.28084
    p0 = 1013.25
    p = p0 * (1 - meters / 44330) ** (1 / 0.1903)
    return p

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

def interpolate_cloud_coverage(pressures, coverage_values, target_altitudes, pressure_altitudes):
    interpolated = []
    for alt_ft in target_altitudes:
        if alt_ft <= pressure_altitudes[0]:
            interpolated.append(coverage_values[0])
        elif alt_ft >= pressure_altitudes[-1]:
            interpolated.append(coverage_values[-1])
        else:
            idx_above = np.searchsorted(pressure_altitudes, alt_ft, side='left')
            idx_below = idx_above - 1
            alt_below = pressure_altitudes[idx_below]
            alt_above = pressure_altitudes[idx_above]
            val_below = coverage_values[idx_below]
            val_above = coverage_values[idx_above]
            frac = (alt_ft - alt_below) / (alt_above - alt_below)
            val = val_below + frac * (val_above - val_below)
            interpolated.append(val)
    return interpolated

def fetch_elevations(points):
    BATCH_SIZE = 100
    elevations = []
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        locations = [{"latitude": lat, "longitude": lon} for lat, lon in batch]
        response = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations}
        )
        if response.status_code == 200:
            results = response.json()['results']
            elevations.extend(r['elevation'] * 3.28084 for r in results)  # meters to feet
        else:
            print(f"Error fetching elevation data: HTTP {response.status_code}")
            elevations.extend([0] * len(batch))
    return elevations

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def miles_to_points(lat1, lon1, lat2, lon2, mile_interval=1.0):
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    return int(distance / mile_interval) + 1

def main(start_lat, start_lon, end_lat, end_lon, grib_file):
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
    pressure_altitudes = np.array([pressure_to_feet(p) for p in pressures])

    num_points = miles_to_points(start_lat, start_lon, end_lat, end_lon)
    print(f"Interpolating {num_points} points along route...")
    route_points = interpolate_lat_lon(start_lat, start_lon, end_lat, end_lon, num_points)

    print("Fetching high-res elevation data...")
    elevations = fetch_elevations(route_points)

    cloud_cache = {}
    cloud_columns = []

    print("Fetching cloud data for unique grid points...")
    for lat, lon in route_points:
        lon_360 = lon % 360
        lat_idx = nearest_index(lat_vals, lat)
        lon_idx = nearest_index(lon_vals, lon_360)
        grid_key = (lat_idx, lon_idx)

        if grid_key not in cloud_cache:
            tcc_slice = ds['tcc'].isel(latitude=lat_idx, longitude=lon_idx)
            cloud_cache[grid_key] = tcc_slice.values

        cloud_columns.append(cloud_cache[grid_key])

    fixed_altitudes = np.arange(0, 12001, 500)
    tcc_columns_interpolated = [
        interpolate_cloud_coverage(pressures, col, fixed_altitudes, pressure_altitudes)
        for col in cloud_columns
    ]
    tcc_by_altitude = list(zip(*tcc_columns_interpolated))

    # print("\nCloud Coverage Graph (x = location, y = altitude):")
    # for row_idx, alt_ft in reversed(list(enumerate(fixed_altitudes))):
    #     row = tcc_by_altitude[row_idx]
    #     bars = "".join(colored_block(val) for val in row)
    #     print(f"{alt_ft:5.0f} ft: {bars}")
    #
    cloud_array = np.array(tcc_by_altitude) / 100.0

    # Build a custom colormap and alpha mask
    base_cmap = mcolors.LinearSegmentedColormap.from_list("light_greys", [(1,1,1), (0.3,0.3,0.3)])
    norm = Normalize(vmin=0.1, vmax=1)
    rgba_img = base_cmap(norm(cloud_array))
    alpha_mask = (cloud_array >= 0.1).astype(float)
    rgba_img[..., -1] = alpha_mask

    fig, ax = plt.subplots(figsize=(14, 6))
    slate_blue = '#6A7B8C'
    fig.patch.set_facecolor(slate_blue)
    ax.set_facecolor(slate_blue)

    ax.imshow(
        rgba_img,
        origin='lower',
        aspect='auto',
        extent=[0, len(elevations), fixed_altitudes[0], fixed_altitudes[-1]],
        interpolation='nearest'
    )

    x = np.arange(len(elevations))
    ax.fill_between(x, 0, elevations, color='olivedrab', zorder=10)
    ax.plot(x, elevations, label='Ground Elevation', color='saddlebrown', linewidth=2, zorder=11)

    ax.set_title("Cloud Coverage and Ground Elevation Along Route")
    ax.set_xlabel("Route Position")
    ax.set_ylabel("Altitude (ft)")
    ax.set_xlim(0, len(elevations))
    ax.set_ylim(0, 12000)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig("elevation_with_clouds.png")
    print("Saved combined elevation and cloud coverage to elevation_with_clouds.png")

if __name__ == "__main__":
    start = input("Start airport (FAA or ICAO): ").strip()
    end = input("End airport (FAA or ICAO): ").strip()
    forecast_hour = input("Forecast hour (e.g., 36 for 36 hours ahead): ").zfill(3)

    try:
        slat, slon = get_airport_latlon(start)
        elat, elon = get_airport_latlon(end)
        grib_file = get_latest_available_gfs_file(forecast_hour)
        main(slat, slon, elat, elon, grib_file)
    except Exception as e:
        print(f"Error: {e}")
