# app.py
# The Adjusters Weather Warrior / Hail and Wind data sets
#
# UPDATES INCLUDED (latest request):
# ✅ Swaths are NO LONGER axis-aligned “grids” — arrows are laid out in an oriented swath band
#    (sampled along a centerline, then offset perpendicular across the swath width).
# ✅ Legend REMOVED from inside the Folium map and shown OUTSIDE the map,
#    above the Arrow Opacity slider (right-side column).
# ✅ Wind + Hail use TO direction (wind is converted from FROM -> TO).
# ✅ Address field accepts either address OR "lat, lon"
# ✅ DOL matching window: Exact, ±1, ±3, ±5
# ✅ Streamlit width warning fixed (use_container_width -> width="stretch")

import datetime as dt
import gzip
import io
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import folium
import pandas as pd
import requests
import streamlit as st

# Optional libs (MRMS overlay + faster math)
try:
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import xarray as xr  # type: ignore

    NUMPY_OK = True
    MRMS_XARRAY_OK = True
except Exception:
    NUMPY_OK = False
    MRMS_XARRAY_OK = False


APP_TITLE = "The Adjusters Weather Warrior"
APP_SUBTITLE = "Hail and Wind data sets"

USER_AGENT = "adjusters-weather-warrior/1.0 (streamlit)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

NOAA_STORM_EVENTS_INDEX = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OSM_STATIC_MAP_URL = "https://staticmap.openstreetmap.de/staticmap.php"

MTARCHIVE_BASE = "https://mtarchive.geol.iastate.edu"
MTARCHIVE_MESH_DIR_TEMPLATE = MTARCHIVE_BASE + "/{Y:04d}/{M:02d}/{D:02d}/mrms/ncep/MESH/"

DEFAULT_RADIUS_MILES = 10.0
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Swath rendering
ALONG_SPACING_KM = 0.55          # spacing along centerline
CROSS_SPACING_KM = 0.55          # spacing across swath width
ARROW_LENGTH_KM = 0.30
SWATH_HALF_WIDTH_KM = 3.0        # wider = more "swath"
PATH_SAMPLE_EVERY_KM = 0.5

DEFAULT_SWATH_THRESHOLD_IN = 0.00
MAX_SWATH_POINTS = 450

DEFAULT_AUTORUN = "1"


# -----------------------------
# Requests session with retries
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retries = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    except Exception:
        pass
    return s


SESSION = make_session()


# -----------------------------
# Formatting helpers
# -----------------------------
def fmt_date(d: dt.date) -> str:
    return d.strftime("%m/%d/%Y")


def fmt_dt(t: dt.datetime) -> str:
    return t.strftime("%m/%d/%Y %H:%M:%S")


def osm_link(lat: float, lon: float, zoom: int = 18) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map={zoom}/{lat:.6f}/{lon:.6f}"


def analyze_link(lat: float, lon: float, label: str = "") -> str:
    safe_label = quote(label or "")
    return f"?lat={lat:.6f}&lon={lon:.6f}&label={safe_label}&autorun={DEFAULT_AUTORUN}"


def google_maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


# -----------------------------
# Input parsing: address OR "lat, lon"
# -----------------------------
def parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    s = text.strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$", s)
    if not m:
        return None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
    except Exception:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


# -----------------------------
# Distance + geometry helpers
# -----------------------------
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.756
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_miles * math.asin(math.sqrt(a))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_miles(lat1, lon1, lat2, lon2) * 1.609344


def haversine_miles_vec(lat0, lon0, lats, lons):
    if not NUMPY_OK:
        return None
    R = 3958.756
    lat0r = np.radians(lat0)
    lon0r = np.radians(lon0)
    latsr = np.radians(pd.Series(lats).astype(float).values)
    lonsr = np.radians(pd.Series(lons).astype(float).values)
    dlat = latsr - lat0r
    dlon = lonsr - lon0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latsr) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360) % 360


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    r_km = 6371.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d_div_r = distance_km / r_km
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_div_r)
        + math.cos(lat1) * math.sin(d_div_r) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d_div_r) * math.cos(lat1),
        math.cos(d_div_r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def distance_band(miles: float) -> str:
    if miles <= 1:
        return "0–1"
    if miles <= 3:
        return "1–3"
    if miles <= 5:
        return "3–5"
    if miles <= 10:
        return "5–10"
    return ">10"


# -----------------------------
# Geocoding / Reverse geocoding
# -----------------------------
@st.cache_data(show_spinner=False)
def geocode_address(address: str) -> Optional[Dict]:
    params = {"q": address, "format": "jsonv2", "limit": 1}
    r = SESSION.get(NOMINATIM_URL, params=params, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    row = rows[0]
    return {"display_name": row.get("display_name"), "lat": float(row["lat"]), "lon": float(row["lon"])}


@st.cache_data(show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    params = {"lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "format": "jsonv2"}
    r = SESSION.get(NOMINATIM_REVERSE_URL, params=params, timeout=25)
    r.raise_for_status()
    payload = r.json()
    name = payload.get("display_name")
    return str(name) if name else None


# -----------------------------
# NOAA Storm Events hail-only cache
# -----------------------------
@st.cache_data(show_spinner=False)
def stormevents_file_for_year(year: int) -> Optional[str]:
    r = SESSION.get(NOAA_STORM_EVENTS_INDEX, timeout=25)
    r.raise_for_status()
    marker = f"StormEvents_details-ftp_v1.0_d{year}_c"
    for token in r.text.split('"'):
        if marker in token and token.endswith(".csv.gz"):
            return NOAA_STORM_EVENTS_INDEX + token
    return None


def parse_hail_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "EVENT_TYPE" not in df.columns:
        return pd.DataFrame()

    work = df[df["EVENT_TYPE"].astype(str).str.lower() == "hail"].copy()
    if work.empty:
        return work

    begin_dt = work.get("BEGIN_DATE_TIME").astype(str).str.strip()
    event_time = pd.to_datetime(begin_dt, format="%d-%b-%y %H:%M:%S", errors="coerce")
    missing = event_time.isna()
    if missing.any():
        event_time.loc[missing] = pd.to_datetime(begin_dt.loc[missing], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    work["event_time"] = event_time

    work["lat"] = pd.to_numeric(work.get("BEGIN_LAT"), errors="coerce")
    work["lon"] = pd.to_numeric(work.get("BEGIN_LON"), errors="coerce")
    end_lat = pd.to_numeric(work.get("END_LAT"), errors="coerce")
    end_lon = pd.to_numeric(work.get("END_LON"), errors="coerce")
    work["lat"] = work["lat"].where(work["lat"].notna() & (work["lat"] != 0), end_lat)
    work["lon"] = work["lon"].where(work["lon"].notna() & (work["lon"] != 0), end_lon)

    keep_cols = [
        "event_time",
        "lat",
        "lon",
        "STATE",
        "CZ_NAME_STR",
        "EVENT_ID",
        "MAGNITUDE",
        "MAGNITUDE_TYPE",
        "BEGIN_LOCATION",
    ]
    for c in keep_cols:
        if c not in work.columns:
            work[c] = None

    work = work.dropna(subset=["event_time", "lat", "lon"])
    return work[keep_cols].copy()


def hail_cache_path_parquet(year: int) -> Path:
    return CACHE_DIR / f"hail_{year}.parquet"


def hail_cache_path_csv(year: int) -> Path:
    return CACHE_DIR / f"hail_{year}.csv"


def load_hail_year_fast(year: int) -> pd.DataFrame:
    pq = hail_cache_path_parquet(year)
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            try:
                pq.unlink()
            except Exception:
                pass

    csvp = hail_cache_path_csv(year)
    if csvp.exists():
        try:
            return pd.read_csv(csvp, parse_dates=["event_time"])
        except Exception:
            try:
                csvp.unlink()
            except Exception:
                pass

    url = stormevents_file_for_year(year)
    if not url:
        return pd.DataFrame()

    src_df = pd.read_csv(url, compression="gzip", low_memory=False)
    hail_df = parse_hail_rows(src_df)
    if hail_df.empty:
        return hail_df

    try:
        hail_df.to_parquet(pq, index=False)
    except Exception:
        hail_df.to_csv(csvp, index=False)

    return hail_df


@st.cache_data(show_spinner=False)
def load_hail_events_two_years_window(reference_date: dt.date) -> pd.DataFrame:
    end = reference_date
    start = end - dt.timedelta(days=730)
    years_needed = sorted({start.year, end.year, end.year - 1})

    frames: List[pd.DataFrame] = []
    for y in years_needed:
        df = load_hail_year_fast(y)
        if df.empty:
            continue
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_hail = pd.concat(frames, ignore_index=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end + dt.timedelta(days=1))
    return all_hail[(all_hail["event_time"] >= start_ts) & (all_hail["event_time"] < end_ts)].copy()


def filter_within_radius(all_hail: pd.DataFrame, lat: float, lon: float, radius_miles: float) -> pd.DataFrame:
    if all_hail.empty:
        return pd.DataFrame()
    work = all_hail.copy()
    if NUMPY_OK:
        work["distance_miles"] = haversine_miles_vec(lat, lon, work["lat"].values, work["lon"].values)
    else:
        work["distance_miles"] = work.apply(
            lambda r: haversine_miles(lat, lon, float(r["lat"]), float(r["lon"])), axis=1
        )
    within = work[work["distance_miles"] <= radius_miles].copy()
    if within.empty:
        return within
    within["distance_band"] = within["distance_miles"].apply(distance_band)
    return within


def dol_matches_from_within(within: pd.DataFrame, loss_date: dt.date, day_window: int) -> pd.DataFrame:
    if within.empty:
        return pd.DataFrame()
    loss_ts = pd.Timestamp(loss_date)
    start_ts = loss_ts - pd.Timedelta(days=day_window)
    end_ts = loss_ts + pd.Timedelta(days=day_window)

    day_ts = within["event_time"].dt.normalize()
    out = within[(day_ts >= start_ts) & (day_ts <= end_ts)].copy()
    if out.empty:
        return out

    out["day_delta"] = (out["event_time"].dt.normalize() - loss_ts).abs().dt.days
    out = out.sort_values(by=["day_delta", "distance_miles", "event_time"], ascending=[True, True, False])
    return out


# -----------------------------
# Hail size helpers
# -----------------------------
def format_hail_size_row(mag, mag_type) -> str:
    try:
        if pd.isna(mag):
            return "—"
        v = float(mag)
    except Exception:
        return "—"

    unit = str(mag_type or "").strip().lower()
    if unit in {"in", "inch", "inches"}:
        return f'{v:g}"'
    if unit:
        return f"{v:g} {mag_type}".strip()
    return f'{v:g}" (reported)'


def max_reported_hail_size_callout(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    if df is None or df.empty:
        return None
    mag = pd.to_numeric(df.get("MAGNITUDE"), errors="coerce")
    work = df.copy()
    work["_mag_num"] = mag
    work = work.dropna(subset=["_mag_num"])
    if work.empty:
        return None

    if "distance_miles" not in work.columns:
        work["distance_miles"] = None
    if "distance_band" not in work.columns:
        work["distance_band"] = None

    work = work.sort_values(by=["_mag_num", "event_time", "distance_miles"], ascending=[False, False, True])
    top = work.iloc[0]

    unit = str(top.get("MAGNITUDE_TYPE") or "").strip().lower()
    if unit in {"in", "inch", "inches"}:
        size_str = f'{float(top["_mag_num"]):g}"'
    elif unit:
        size_str = f"{float(top['_mag_num']):g} {top.get('MAGNITUDE_TYPE')}".strip()
    else:
        size_str = f'{float(top["_mag_num"]):g}" (reported)'

    ts = top.get("event_time")
    try:
        ts_str = pd.to_datetime(ts).strftime("%m/%d/%Y %H:%M:%S")
    except Exception:
        ts_str = str(ts)

    dist = top.get("distance_miles")
    try:
        dist_str = f"{float(dist):.2f} mi"
    except Exception:
        dist_str = "—"

    loc = str(top.get("BEGIN_LOCATION") or "").strip()
    band = str(top.get("distance_band") or "").strip()
    return {"size": size_str, "time": ts_str, "distance": dist_str, "band": band, "location": loc if loc else "—"}


# -----------------------------
# Open-Meteo wind (MPH)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_open_meteo_wind_mph(lat: float, lon: float, day: dt.date) -> Optional[Dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "daily": "windspeed_10m_max,winddirection_10m_dominant",
        "timezone": "auto",
        "windspeed_unit": "mph",
    }
    r = SESSION.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=25)
    r.raise_for_status()
    payload = r.json()
    daily = payload.get("daily", {})
    speeds = daily.get("windspeed_10m_max", [])
    directions = daily.get("winddirection_10m_dominant", [])
    if not speeds or not directions:
        return None
    if speeds[0] is None or directions[0] is None:
        return None
    return {"wind_speed_max_mph": float(speeds[0]), "wind_dir_dominant": float(directions[0])}


def wind_from_to_dir(wind_dir_from: float) -> float:
    return (float(wind_dir_from) + 180.0) % 360.0


# -----------------------------
# MRMS MESH via MTArchive (overlay + swath points)
# -----------------------------
@st.cache_data(show_spinner=False)
def list_mtarchive_grib_files_for_day(day: dt.date) -> List[Tuple[dt.datetime, str]]:
    dir_url = MTARCHIVE_MESH_DIR_TEMPLATE.format(Y=day.year, M=day.month, D=day.day)
    r = SESSION.get(dir_url, timeout=30)
    r.raise_for_status()
    hrefs = re.findall(r'href="([^"]+\.grib2\.gz)"', r.text)

    out: List[Tuple[dt.datetime, str]] = []
    for fn in hrefs:
        m = re.search(r"(\d{8})-(\d{6})", fn)
        if not m:
            continue
        ts = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        out.append((ts, dir_url + fn))
    out.sort(key=lambda x: x[0])
    return out


def choose_nearest_timestamp(files: List[Tuple[dt.datetime, str]], target: dt.datetime) -> Optional[Tuple[dt.datetime, str, float]]:
    if not files:
        return None
    best = min(files, key=lambda p: abs((p[0] - target).total_seconds()))
    delta_min = abs((best[0] - target).total_seconds()) / 60.0
    return best[0], best[1], delta_min


def parse_mesh_with_xarray(grib_path: str):
    if not MRMS_XARRAY_OK:
        return None
    try:
        return xr.open_dataset(grib_path, engine="cfgrib")
    except Exception:
        return None


def sample_mesh_nearest_value(ds, lat: float, lon: float) -> Optional[float]:
    if ds is None or not getattr(ds, "data_vars", None):
        return None
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
    lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
    if not lat_name or not lon_name:
        return None
    try:
        sel = da.sel({lat_name: lat, lon_name: lon}, method="nearest")
        val = float(sel.values)
        if NUMPY_OK and np.isnan(val):
            return None
        return val
    except Exception:
        return None


def make_mesh_overlay_png_and_bounds(ds, center_lat: float, center_lon: float, box_deg: float = 0.12) -> Optional[Tuple[bytes, List[List[float]]]]:
    if not (MRMS_XARRAY_OK and NUMPY_OK) or ds is None:
        return None
    if not getattr(ds, "data_vars", None):
        return None

    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
    lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
    if not lat_name or not lon_name:
        return None

    lat_min, lat_max = center_lat - box_deg, center_lat + box_deg
    lon_min, lon_max = center_lon - box_deg, center_lon + box_deg

    try:
        sub = da.where(
            (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max)
            & (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max),
            drop=True,
        )
        if sub.size == 0:
            return None

        lat_s = float(sub[lat_name].min().values)
        lat_n = float(sub[lat_name].max().values)
        lon_w = float(sub[lon_name].min().values)
        lon_e = float(sub[lon_name].max().values)
        bounds = [[lat_s, lon_w], [lat_n, lon_e]]

        arr = sub.values
        masked = np.ma.masked_where(np.isnan(arr) | (arr <= 0), arr)

        fig = plt.figure(figsize=(6, 6), dpi=140)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.imshow(masked, interpolation="nearest", aspect="auto")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return buf.getvalue(), bounds
    except Exception:
        return None


def mesh_swath_points_from_ds(
    ds,
    center_lat: float,
    center_lon: float,
    threshold_in: float,
    box_deg: float = 0.12,
    max_points: int = MAX_SWATH_POINTS,
) -> Optional[List[Tuple[float, float]]]:
    if ds is None or not getattr(ds, "data_vars", None) or not NUMPY_OK:
        return None

    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
    lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)
    if not lat_name or not lon_name:
        return None

    lat_min, lat_max = center_lat - box_deg, center_lat + box_deg
    lon_min, lon_max = center_lon - box_deg, center_lon + box_deg

    try:
        sub = da.where(
            (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max)
            & (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max),
            drop=True,
        )
        if sub.size == 0:
            return None

        vals_in = sub.values / 25.4
        mask = np.isfinite(vals_in) & (vals_in >= float(threshold_in)) & (vals_in > 0)
        if not mask.any():
            return []

        latc = sub[lat_name].values
        lonc = sub[lon_name].values

        yy, xx = np.where(mask)
        if latc.ndim == 1 and lonc.ndim == 1:
            pts = list(zip(latc[yy].astype(float), lonc[xx].astype(float)))
        else:
            pts = list(zip(latc[yy, xx].astype(float), lonc[yy, xx].astype(float)))

        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
            pts = [pts[i] for i in idx]
        return pts
    except Exception:
        return None


def fetch_mrms_mesh_hail_size_inches(lat: float, lon: float, hail_dt: dt.datetime, threshold_in: float) -> Dict[str, Optional[object]]:
    result: Dict[str, Optional[object]] = {
        "available": False,
        "mesh_size_in": None,
        "product_time_utc": None,
        "distance_minutes_from_hail_time": None,
        "source_url": None,
        "overlay_png": None,
        "overlay_bounds": None,
        "swath_points": None,
        "status": "MRMS MESH unavailable.",
    }

    try:
        files = list_mtarchive_grib_files_for_day(hail_dt.date())
    except Exception as exc:
        result["status"] = f"MTArchive request failed: {exc}"
        return result

    choice = choose_nearest_timestamp(files, hail_dt)
    if not choice:
        result["status"] = "No MRMS MESH files listed for that date."
        return result

    file_dt, url, delta_min = choice
    result["available"] = True
    result["product_time_utc"] = file_dt
    result["distance_minutes_from_hail_time"] = round(delta_min, 1)
    result["source_url"] = url

    if not MRMS_XARRAY_OK:
        result["status"] = "MRMS file located, but parsing packages not installed (xarray/cfgrib/eccodes/matplotlib)."
        return result

    try:
        rr = SESSION.get(url, timeout=70)
        rr.raise_for_status()
        grib_bytes = gzip.decompress(rr.content)

        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(grib_bytes)
            grib_path = tmp.name

        try:
            ds = parse_mesh_with_xarray(grib_path)
            if ds is None:
                result["status"] = "MRMS file downloaded but could not be parsed (cfgrib/eccodes issue)."
                return result

            mesh_mm = sample_mesh_nearest_value(ds, lat, lon)
            if mesh_mm is not None:
                inches = float(mesh_mm) / 25.4
                if math.isfinite(inches):
                    result["mesh_size_in"] = max(0.0, inches)

            overlay = make_mesh_overlay_png_and_bounds(ds, lat, lon, box_deg=0.12)
            if overlay:
                result["overlay_png"], result["overlay_bounds"] = overlay

            result["swath_points"] = mesh_swath_points_from_ds(ds, lat, lon, threshold_in=threshold_in, box_deg=0.12)
            result["status"] = "MRMS MESH parsed successfully."
            return result
        finally:
            try:
                os.remove(grib_path)
            except OSError:
                pass
    except Exception as exc:
        result["status"] = f"MRMS download/parse failed: {exc}"
        return result


# -----------------------------
# NOAA points -> ordered path + densify
# -----------------------------
def noaa_report_points(df: pd.DataFrame, max_points: int = 250) -> List[Tuple[float, float]]:
    if df is None or df.empty:
        return []
    pts: List[Tuple[float, float]] = []
    for _, r in df.iterrows():
        try:
            pts.append((float(r["lat"]), float(r["lon"])))
        except Exception:
            continue

    seen = set()
    uniq: List[Tuple[float, float]] = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    if len(uniq) <= max_points:
        return uniq

    step = max(1, len(uniq) // max_points)
    return uniq[::step]


def order_points_pathlike(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return points
    if NUMPY_OK:
        arr = np.array([[p[0], p[1]] for p in points], dtype=float)
        mean = arr.mean(axis=0)
        X = arr - mean
        cov = (X.T @ X) / max(1, (len(points) - 1))
        vals, vecs = np.linalg.eig(cov)
        axis = vecs[:, int(np.argmax(vals))]
        proj = X @ axis
        idx = np.argsort(proj)
        return [points[int(i)] for i in idx]
    return points


def densify_polyline(points: List[Tuple[float, float]], sample_every_km: float = PATH_SAMPLE_EVERY_KM) -> List[Tuple[float, float]]:
    if len(points) <= 1:
        return points
    out: List[Tuple[float, float]] = [points[0]]
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        seg_km = haversine_km(a[0], a[1], b[0], b[1])
        if seg_km <= 0.001:
            continue
        steps = max(1, int(seg_km / sample_every_km))
        for s in range(1, steps + 1):
            t = s / (steps + 1)
            lat = a[0] + (b[0] - a[0]) * t
            lon = a[1] + (b[1] - a[1]) * t
            out.append((lat, lon))
        out.append(b)
    return out


# -----------------------------
# Map arrows (swath band, not grid)
# -----------------------------
def add_quiver_like_arrow(
    fg: folium.FeatureGroup,
    start: Tuple[float, float],
    end: Tuple[float, float],
    bearing_deg: float,
    color: str,
    opacity: float,
    weight: int = 2,
) -> None:
    (lat1, lon1) = start
    (lat2, lon2) = end

    folium.PolyLine(locations=[(lat1, lon1), (lat2, lon2)], color=color, weight=weight, opacity=opacity).add_to(fg)

    head_len_km = 0.12
    head_angle = 35.0
    left_bearing = (bearing_deg + 180 - head_angle) % 360
    right_bearing = (bearing_deg + 180 + head_angle) % 360

    l_end = destination_point(lat2, lon2, left_bearing, head_len_km)
    r_end = destination_point(lat2, lon2, right_bearing, head_len_km)

    folium.PolyLine(locations=[(lat2, lon2), l_end], color=color, weight=weight, opacity=opacity).add_to(fg)
    folium.PolyLine(locations=[(lat2, lon2), r_end], color=color, weight=weight, opacity=opacity).add_to(fg)


def sample_polyline_by_distance(points: List[Tuple[float, float]], step_km: float) -> List[Tuple[float, float]]:
    """
    Resample polyline so points are roughly step_km apart along-track.
    """
    if len(points) < 2:
        return points
    out = [points[0]]
    acc = 0.0
    last = points[0]
    for i in range(1, len(points)):
        cur = points[i]
        seg = haversine_km(last[0], last[1], cur[0], cur[1])
        if seg <= 1e-6:
            continue
        while acc + seg >= step_km:
            remain = step_km - acc
            t = remain / seg
            lat = last[0] + (cur[0] - last[0]) * t
            lon = last[1] + (cur[1] - last[1]) * t
            out.append((lat, lon))
            # advance
            last = (lat, lon)
            seg = haversine_km(last[0], last[1], cur[0], cur[1])
            acc = 0.0
        acc += seg
        last = cur
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def add_oriented_swath_band(
    fg: folium.FeatureGroup,
    centerline: List[Tuple[float, float]],
    to_bearing_deg: float,
    color: str,
    opacity: float,
    along_spacing_km: float,
    cross_spacing_km: float,
    half_width_km: float,
    arrow_len_km: float,
) -> None:
    """
    Oriented swath:
      - march along the centerline (along_spacing_km)
      - at each sample, place arrows across a perpendicular cross-section
      - arrow direction is to_bearing_deg (TO direction)
    """
    if len(centerline) < 2:
        return

    # resample so we get consistent along spacing
    samples = sample_polyline_by_distance(centerline, along_spacing_km)

    # perpendicular direction for cross offsets
    left_dir = (to_bearing_deg - 90.0) % 360.0
    right_dir = (to_bearing_deg + 90.0) % 360.0

    # offsets from -half_width..+half_width
    n = int((2 * half_width_km) / cross_spacing_km) + 1
    offsets = [(-half_width_km + i * cross_spacing_km) for i in range(n)]

    for (lat, lon) in samples:
        for off in offsets:
            if abs(off) < 1e-9:
                o_lat, o_lon = lat, lon
            elif off < 0:
                o_lat, o_lon = destination_point(lat, lon, left_dir, abs(off))
            else:
                o_lat, o_lon = destination_point(lat, lon, right_dir, abs(off))

            end_lat, end_lon = destination_point(o_lat, o_lon, to_bearing_deg, arrow_len_km)
            add_quiver_like_arrow(
                fg=fg,
                start=(o_lat, o_lon),
                end=(end_lat, end_lon),
                bearing_deg=to_bearing_deg,
                color=color,
                opacity=opacity,
                weight=2,
            )


def hail_centerline_from_points(report_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not report_points:
        return []
    ordered = order_points_pathlike(report_points)
    dense = densify_polyline(ordered, sample_every_km=PATH_SAMPLE_EVERY_KM)
    return dense


def fetch_static_map_image(lat: float, lon: float, zoom: int = 18, width: int = 800, height: int = 500) -> Optional[bytes]:
    params = {
        "center": f"{lat:.6f},{lon:.6f}",
        "zoom": str(zoom),
        "size": f"{width}x{height}",
        "markers": f"{lat:.6f},{lon:.6f},lightblue1",
    }
    r = SESSION.get(OSM_STATIC_MAP_URL, params=params, timeout=30)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "").lower()
    if "image" not in content_type:
        return None
    return r.content


def build_map(
    prop_lat: float,
    prop_lon: float,
    wind_to_dir: Optional[float],
    mesh: Optional[Dict],
    hail_points_reported: Optional[List[Tuple[float, float]]],
    show_wind: bool,
    show_hail: bool,
    arrows_opacity: float,
) -> folium.Map:
    m = folium.Map(location=[prop_lat, prop_lon], zoom_start=16, control_scale=True)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.Marker([prop_lat, prop_lon], tooltip="Property").add_to(m)

    wind_fg = folium.FeatureGroup(name="Wind swath (blue)", show=show_wind)
    hail_fg = folium.FeatureGroup(name="Hail swath (yellow)", show=show_hail)

    # hail base points (MRMS preferred)
    hail_base_points: List[Tuple[float, float]] = []
    if mesh and mesh.get("swath_points"):
        hail_base_points = list(mesh["swath_points"] or [])
    elif hail_points_reported:
        hail_base_points = list(hail_points_reported or [])

    hail_centerline = hail_centerline_from_points(hail_base_points)

    # HAIL swath direction: average centerline segment bearing (TO direction)
    hail_to_dir: Optional[float] = None
    if hail_centerline and len(hail_centerline) >= 2:
        seg_bearings: List[float] = []
        for i in range(len(hail_centerline) - 1):
            a = hail_centerline[i]
            b = hail_centerline[i + 1]
            if haversine_km(a[0], a[1], b[0], b[1]) < 0.08:
                continue
            seg_bearings.append(bearing_between_points(a[0], a[1], b[0], b[1]))
        if seg_bearings:
            hail_to_dir = float(np.mean(seg_bearings)) if NUMPY_OK else float(seg_bearings[0])

    # HAIL swath band
    if show_hail and hail_centerline and hail_to_dir is not None:
        add_oriented_swath_band(
            fg=hail_fg,
            centerline=hail_centerline,
            to_bearing_deg=hail_to_dir,
            color="#f1c40f",
            opacity=float(arrows_opacity),
            along_spacing_km=ALONG_SPACING_KM,
            cross_spacing_km=CROSS_SPACING_KM,
            half_width_km=SWATH_HALF_WIDTH_KM,
            arrow_len_km=ARROW_LENGTH_KM,
        )

    # WIND swath band:
    # If hail centerline exists, match the same footprint for comparison; else use a short local line around property.
    if show_wind and wind_to_dir is not None:
        if hail_centerline and len(hail_centerline) >= 2:
            wind_centerline = hail_centerline
        else:
            # small fallback centerline
            wind_centerline = [
                destination_point(prop_lat, prop_lon, (wind_to_dir + 180) % 360, 1.0),
                destination_point(prop_lat, prop_lon, wind_to_dir, 1.0),
            ]

        add_oriented_swath_band(
            fg=wind_fg,
            centerline=wind_centerline,
            to_bearing_deg=float(wind_to_dir),
            color="#1f77b4",
            opacity=float(arrows_opacity),
            along_spacing_km=ALONG_SPACING_KM,
            cross_spacing_km=CROSS_SPACING_KM,
            half_width_km=SWATH_HALF_WIDTH_KM,
            arrow_len_km=ARROW_LENGTH_KM,
        )

    wind_fg.add_to(m)
    hail_fg.add_to(m)

    # MRMS overlay image
    if mesh and mesh.get("overlay_png") and mesh.get("overlay_bounds"):
        tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_png.write(mesh["overlay_png"])
        tmp_png.close()
        folium.raster_layers.ImageOverlay(
            name="MRMS MESH overlay (radar area)",
            image=tmp_png.name,
            bounds=mesh["overlay_bounds"],
            opacity=0.45,
            interactive=True,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# -----------------------------
# PDF report (optional)
# -----------------------------
def wrap_line(text: str, width: int = 95) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    count = 0
    for w in words:
        new_count = count + len(w) + (1 if current else 0)
        if new_count > width:
            lines.append(" ".join(current))
            current = [w]
            count = len(w)
        else:
            current.append(w)
            count = new_count
    if current:
        lines.append(" ".join(current))
    return lines


def build_pdf_report(
    address: str,
    lat: float,
    lon: float,
    generated_ts_utc: dt.datetime,
    radius_miles: float,
    selected_mode: str,
    dol: Optional[dt.date],
    dol_window_label: Optional[str],
    wind: Optional[Dict],
    mesh: Optional[Dict],
    swath_threshold_in: float,
    map_url: str,
    map_image_bytes: Optional[bytes] = None,
) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError("ReportLab is required for PDF generation.") from exc

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    def write_line(line: str, step: int = 15) -> None:
        nonlocal y
        c.drawString(50, y, line)
        y -= step

    c.setFont("Helvetica-Bold", 14)
    write_line(f"{APP_TITLE} — {APP_SUBTITLE}", step=22)
    c.setFont("Helvetica", 10)

    write_line(f"Address/Location: {address}")
    write_line(f"Coordinates: {lat:.6f}, {lon:.6f}")
    write_line(f"NOAA hail radius: {radius_miles:.1f} miles")
    write_line(f"Report generated (UTC): {generated_ts_utc.strftime('%m/%d/%Y %H:%M:%S')}")
    y -= 5

    write_line("Mode / selection:")
    write_line(f"- Mode: {selected_mode}")
    if dol:
        write_line(f"- Claimed DOL: {fmt_date(dol)} ({dol_window_label or ''})")
    y -= 5

    write_line("Wind / hail sizing:")
    if wind:
        write_line(f"- Wind max (10m): {wind['wind_speed_max_mph']:.1f} MPH")
        write_line(f"- Wind dominant direction (FROM): {wind['wind_dir_dominant']:.0f}°")
        write_line(f"- Wind TO direction: {wind_from_to_dir(wind['wind_dir_dominant']):.0f}°")
    else:
        write_line("- Wind: unavailable for selected date")

    if mesh and mesh.get("available"):
        if mesh.get("mesh_size_in") is not None:
            write_line(f"- MRMS MESH (radar-est.) @ property: {float(mesh['mesh_size_in']):.2f} inches")
        else:
            write_line("- MRMS MESH (radar-est.) @ property: unavailable")
        write_line(f"- MRMS swath threshold: {swath_threshold_in:.2f} inches")
        for seg in wrap_line(f"- MRMS status: {mesh.get('status','')}"):
            write_line(seg)
    else:
        write_line("- MRMS MESH: unavailable")
    y -= 5

    if map_image_bytes:
        try:
            image_reader = ImageReader(io.BytesIO(map_image_bytes))
            max_w = width - 100
            map_h = 220
            if y < (map_h + 80):
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
            write_line("Map snapshot:")
            c.drawImage(image_reader, 50, y - map_h, width=max_w, height=map_h, preserveAspectRatio=True, mask="auto")
            y -= (map_h + 12)
        except Exception:
            write_line("Map link:")
            for seg in wrap_line(map_url):
                write_line(seg)
            y -= 8
    else:
        write_line("Map link:")
        for seg in wrap_line(map_url):
            write_line(seg)
        y -= 8

    c.setFont("Helvetica-Oblique", 10)
    disclaimer = (
        "Disclaimer: This report summarizes publicly available meteorological datasets. "
        "Direction arrows show TO direction for inspection planning only and do not confirm physical damage or causation."
    )
    for seg in wrap_line(disclaimer):
        write_line(seg)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


# -----------------------------
# UI helpers
# -----------------------------
def init_state() -> None:
    if "has_result" not in st.session_state:
        st.session_state.has_result = False
    if "result" not in st.session_state:
        st.session_state.result = {}
    if "arrows_opacity" not in st.session_state:
        st.session_state.arrows_opacity = 0.75


def clear_form() -> None:
    st.session_state.has_result = False
    st.session_state.result = {}
    st.session_state.address = ""
    st.session_state.radius_miles = int(DEFAULT_RADIUS_MILES)
    st.session_state.use_dol = True
    st.session_state.show_wind = True
    st.session_state.show_hail = True
    st.session_state.arrows_opacity = 0.75
    st.session_state.dol = dt.date.today()
    st.session_state.dol_window_label = "±1 day"
    st.session_state.hail_swath_threshold_in = float(DEFAULT_SWATH_THRESHOLD_IN)


def inject_center_table_css() -> None:
    st.markdown(
        """
<style>
div[data-testid="stDataFrame"] table td,
div[data-testid="stDataFrame"] table th {
  text-align: center !important;
}
div[data-testid="stDataEditor"] table td,
div[data-testid="stDataEditor"] table th {
  text-align: center !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def read_query_params() -> Dict[str, str]:
    try:
        qp = st.query_params
        out = {}
        for k in qp.keys():
            v = qp.get(k)
            if isinstance(v, list):
                out[k] = v[0] if v else ""
            else:
                out[k] = str(v) if v is not None else ""
        return out
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            return {k: (v[0] if isinstance(v, list) and v else str(v)) for k, v in qp.items()}
        except Exception:
            return {}


def legend_block_html() -> str:
    return """
<div style="
border:1px solid #ddd;
border-radius:12px;
padding:12px 12px;
background:#fff;
box-shadow:0 1px 8px rgba(0,0,0,0.08);
font-size:12px;
">
  <div style="font-weight:700; margin-bottom:8px;">Weather Warrior legend</div>
  <div style="margin-bottom:6px;"><b>Direction swaths (TO direction)</b></div>
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="display:inline-block; width:14px; height:14px; background:#1f77b4; border-radius:3px;"></span>
    <span>Wind swath (blue)</span>
  </div>
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
    <span style="display:inline-block; width:14px; height:14px; background:#f1c40f; border-radius:3px;"></span>
    <span>Hail swath (yellow)</span>
  </div>
  <div style="color:#555;">
    Inspection planning only; does not confirm damage.
  </div>
</div>
"""


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    init_state()
    inject_center_table_css()

    st.set_page_config(page_title=f"{APP_TITLE} — {APP_SUBTITLE}", layout="wide"), initial_sidebar_state="collapsed"
    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.caption(
        "Neutral hail proximity + wind direction + MRMS MESH context. "
        "All outputs are for inspection planning only and do not confirm physical damage."
    )

    qp = read_query_params()
    qp_lat = qp.get("lat", "").strip()
    qp_lon = qp.get("lon", "").strip()
    qp_label = qp.get("label", "").strip()
    qp_autorun = qp.get("autorun", "").strip()

    with st.expander("NOAA performance controls (recommended)", expanded=False):
        st.write(f"Cache folder: **{CACHE_DIR.resolve()}**")
        cached_files = sorted(CACHE_DIR.glob("hail_*.parquet")) + sorted(CACHE_DIR.glob("hail_*.csv"))
        st.write(f"Cached hail-year files found: **{len(cached_files)}**")
        preload_date = st.date_input("Preload for reference date", value=dt.date.today(), key="preload_date")
        if st.button("Preload NOAA hail cache now"):
            with st.spinner("Preloading NOAA hail cache (one-time per needed year)..."):
                _ = load_hail_events_two_years_window(preload_date)
            st.success("Preload complete. Future searches should be much faster.")

    address = st.text_input("Property address OR lat, lon (example: 37.12345, -84.98765)", key="address")

    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1])
    with c1:
        radius_miles = float(st.slider("NOAA hail search radius (miles)", 1, 25, int(DEFAULT_RADIUS_MILES), key="radius_miles"))
    with c2:
        use_dol = st.checkbox("Use Date of Loss (DOL)", value=True, key="use_dol")
    with c3:
        show_wind = st.checkbox("Wind swath (blue)", value=True, key="show_wind")
    with c4:
        show_hail = st.checkbox("Hail swath (yellow)", value=True, key="show_hail")

    hail_swath_threshold_in = st.slider(
        "Hail swath threshold (MRMS MESH, inches)",
        min_value=0.00,
        max_value=3.00,
        value=float(st.session_state.get("hail_swath_threshold_in", DEFAULT_SWATH_THRESHOLD_IN)),
        step=0.05,
        key="hail_swath_threshold_in",
        help="Hail swath uses MRMS cells where MESH >= this size. If MRMS is unavailable, NOAA-reported points are used.",
    )

    dol = st.date_input("Claimed Date of Loss", value=dt.date.today(), disabled=not use_dol, key="dol")

    dol_window_label = st.selectbox(
        "DOL matching window",
        options=["Exact", "±1 day", "±3 days", "±5 days"],
        index=1,
        disabled=not use_dol,
        key="dol_window_label",
    )
    window_map = {"Exact": 0, "±1 day": 1, "±3 days": 3, "±5 days": 5}
    dol_window_days = window_map[dol_window_label] if use_dol else 0

    run = st.button("Analyze", key="analyze_btn")
    autorun = bool(qp_lat) and bool(qp_lon) and (qp_autorun in ("", "1", "true", "True"))
    if autorun:
        run = True

    st.info("Inspection planning only; does not confirm damage.")

    if run:
        try:
            prop_lat = None
            prop_lon = None
            resolved_address = None

            if qp_lat and qp_lon:
                prop_lat = float(qp_lat)
                prop_lon = float(qp_lon)
                with st.spinner("Reverse-geocoding selected NOAA point..."):
                    resolved_address = reverse_geocode(prop_lat, prop_lon) or qp_label or f"{prop_lat:.6f}, {prop_lon:.6f}"
                st.session_state.address = resolved_address

            if prop_lat is None or prop_lon is None:
                parsed = parse_latlon(address)
                if parsed:
                    prop_lat, prop_lon = parsed
                    with st.spinner("Reverse-geocoding coordinates..."):
                        resolved_address = reverse_geocode(prop_lat, prop_lon) or f"{prop_lat:.6f}, {prop_lon:.6f}"
                else:
                    if not address.strip():
                        st.warning("Enter an address or coordinates to continue.")
                        return
                    with st.spinner("Geocoding address..."):
                        geocode = geocode_address(address)
                    if not geocode:
                        st.error("Address not found.")
                        return
                    prop_lat = geocode["lat"]
                    prop_lon = geocode["lon"]
                    resolved_address = geocode["display_name"] or address

            assert prop_lat is not None and prop_lon is not None

            ref_date = dol if use_dol else dt.date.today()
            try:
                with st.spinner("Loading NOAA hail events (cached hail-only; first run may take longer)..."):
                    all_hail = load_hail_events_two_years_window(ref_date)
            except Exception as exc:
                st.warning(
                    "NOAA hail data is unavailable right now (network/DNS/proxy issue). "
                    "You can still use Wind + MRMS + Map.\n\n"
                    f"Details: {exc}"
                )
                all_hail = pd.DataFrame()

            within = filter_within_radius(all_hail, prop_lat, prop_lon, radius_miles) if not all_hail.empty else pd.DataFrame()

            dol_matches = pd.DataFrame()
            if use_dol and not within.empty:
                dol_matches = dol_matches_from_within(within, dol, int(dol_window_days))

            active_df = dol_matches if (use_dol and not dol_matches.empty) else within

            selected_mode = "Selected event"
            analysis_dt: Optional[dt.datetime] = None

            if use_dol and not dol_matches.empty:
                selected_mode = "DOL match (nearest in window)"
                analysis_dt = dol_matches.iloc[0]["event_time"].to_pydatetime()
            elif not within.empty:
                analysis_dt = within.sort_values("event_time", ascending=False).iloc[0]["event_time"].to_pydatetime()

            if analysis_dt is None:
                st.warning("No hail report timestamp available (within your selected radius).")
                st.session_state.has_result = False
                st.session_state.result = {}
                return

            analysis_date = analysis_dt.date()

            with st.spinner("Querying Open-Meteo archive for wind (MPH)..."):
                wind = fetch_open_meteo_wind_mph(prop_lat, prop_lon, analysis_date)

            with st.spinner("Querying MRMS MESH (inches) and radar overlay..."):
                mesh = fetch_mrms_mesh_hail_size_inches(prop_lat, prop_lon, analysis_dt, threshold_in=float(hail_swath_threshold_in))

            wind_to_dir = None
            if wind and "wind_dir_dominant" in wind:
                wind_to_dir = wind_from_to_dir(float(wind["wind_dir_dominant"]))

            hail_points_reported = noaa_report_points(active_df)

            st.session_state.result = {
                "resolved_address": resolved_address,
                "prop_lat": prop_lat,
                "prop_lon": prop_lon,
                "radius_miles": radius_miles,
                "use_dol": use_dol,
                "dol": dol if use_dol else None,
                "dol_window_label": dol_window_label if use_dol else None,
                "dol_window_days": int(dol_window_days) if use_dol else None,
                "within": within,
                "dol_matches": dol_matches,
                "active_df": active_df,
                "analysis_dt": analysis_dt,
                "selected_mode": selected_mode,
                "wind": wind,
                "mesh": mesh,
                "wind_to_dir": wind_to_dir,
                "hail_points_reported": hail_points_reported,
                "map_url": osm_link(prop_lat, prop_lon),
                "google_maps_url": google_maps_link(prop_lat, prop_lon),
                "hail_swath_threshold_in": float(hail_swath_threshold_in),
            }
            st.session_state.has_result = True

        except requests.RequestException as exc:
            st.error(f"Network/API request failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    if st.session_state.has_result:
        r = st.session_state.result
        st.write(f"**Resolved location:** {r['resolved_address']}")

        st.subheader("NOAA hail reports (last 2 years) within selected radius")
        within = r.get("within", pd.DataFrame())
        if within is None or within.empty:
            st.info("No NOAA hail reports loaded/available for this run (or none within radius).")
        else:
            table = within.sort_values("event_time", ascending=False).copy()
            table["Event Date/Time"] = pd.to_datetime(table["event_time"]).dt.strftime("%m/%d/%Y %H:%M:%S")
            table["Distance (mi)"] = pd.to_numeric(table["distance_miles"], errors="coerce").round(2)
            table["Band (mi)"] = table["distance_band"].astype(str)
            table["Approx Hail Size"] = table.apply(lambda rr: format_hail_size_row(rr.get("MAGNITUDE"), rr.get("MAGNITUDE_TYPE")), axis=1)
            table["Lat"] = pd.to_numeric(table["lat"], errors="coerce").round(5)
            table["Lon"] = pd.to_numeric(table["lon"], errors="coerce").round(5)

            table["Map Link"] = table.apply(lambda rr: osm_link(float(rr["lat"]), float(rr["lon"]), zoom=14), axis=1)
            table["Analyze Link"] = table.apply(
                lambda rr: analyze_link(float(rr["lat"]), float(rr["lon"]), label=str(rr.get("BEGIN_LOCATION") or "")),
                axis=1,
            )

            show = table[
                ["Event Date/Time", "Distance (mi)", "Band (mi)", "Approx Hail Size", "BEGIN_LOCATION", "STATE", "CZ_NAME_STR", "Lat", "Lon", "Map Link", "Analyze Link"]
            ].rename(columns={"BEGIN_LOCATION": "Location"})

            st.data_editor(
                show,
                width="stretch",
                hide_index=True,
                disabled=True,
                column_config={
                    "Map Link": st.column_config.LinkColumn("Map Link", display_text="Open map"),
                    "Analyze Link": st.column_config.LinkColumn("Analyze Link", display_text="Analyze here"),
                },
            )

        st.subheader("Approx Hail size + Wind speed")
        left, right = st.columns(2)

        with left:
            st.markdown("#### Hail")
            st.write(f"Analysis timestamp: **{fmt_dt(r['analysis_dt'])}**")
            st.write(f"Mode: **{r['selected_mode']}**")

            active_df = r.get("active_df", pd.DataFrame())
            callout = max_reported_hail_size_callout(active_df) if active_df is not None else None
            if callout:
                st.success(
                    f'**Largest reported size in window:** {callout["size"]}  '
                    f'(Time: {callout["time"]}, Distance: {callout["distance"]} '
                    f'{("[" + callout["band"] + "]") if callout["band"] else ""}, '
                    f'Location: {callout["location"]})'
                )
            else:
                st.info("Largest reported size in window: unavailable.")

            mesh = r.get("mesh")
            if mesh and mesh.get("available"):
                if mesh.get("mesh_size_in") is not None:
                    st.write(f'MRMS MESH (radar-est.) @ property: **{float(mesh["mesh_size_in"]):.2f} inches**')
                else:
                    st.write("MRMS MESH (radar-est.) @ property: unavailable.")
                st.write(f'MRMS hail swath threshold: **{float(r.get("hail_swath_threshold_in", DEFAULT_SWATH_THRESHOLD_IN)):.2f} inches**')
                st.write(f'MRMS status: {mesh.get("status","N/A")}')
            else:
                st.write("MRMS MESH: unavailable. (Hail swath uses NOAA reported points.)")

        with right:
            st.markdown("#### Wind")
            wind = r.get("wind")
            if wind:
                st.write(f'Wind max (10m): **{wind["wind_speed_max_mph"]:.1f} MPH**')
                st.write(f'Dominant wind direction (FROM): **{wind["wind_dir_dominant"]:.0f}°**')
                if r.get("wind_to_dir") is not None:
                    st.write(f'Wind TO direction: **{float(r["wind_to_dir"]):.0f}°**')
            else:
                st.write("Wind data: unavailable.")

        st.subheader("Weather radar area map")
        map_col, ctl_col = st.columns([0.82, 0.18], vertical_alignment="bottom")

        with map_col:
            m2 = build_map(
                prop_lat=r["prop_lat"],
                prop_lon=r["prop_lon"],
                wind_to_dir=r.get("wind_to_dir"),
                mesh=r.get("mesh"),
                hail_points_reported=r.get("hail_points_reported"),
                show_wind=st.session_state.get("show_wind", True),
                show_hail=st.session_state.get("show_hail", True),
                arrows_opacity=float(st.session_state.get("arrows_opacity", 0.75)),
            )
            st.components.v1.html(m2._repr_html_(), height=650)
            st.markdown(f"[Open this location in Google Maps]({r.get('google_maps_url')})")

        with ctl_col:
            # ✅ Legend outside map, above slider
            st.markdown(legend_block_html(), unsafe_allow_html=True)
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            st.markdown("##### Map controls")
            st.slider(
                "Arrow opacity",
                min_value=0.05,
                max_value=1.0,
                value=float(st.session_state.get("arrows_opacity", 0.75)),
                step=0.05,
                key="arrows_opacity",
                help="Lower = more transparent. Higher = more solid.",
            )

        st.subheader("Export")
        generated_ts = dt.datetime.now(dt.timezone.utc)

        map_image_bytes: Optional[bytes] = None
        try:
            map_image_bytes = fetch_static_map_image(r["prop_lat"], r["prop_lon"])
        except Exception:
            map_image_bytes = None

        try:
            pdf_bytes = build_pdf_report(
                address=r["resolved_address"],
                lat=r["prop_lat"],
                lon=r["prop_lon"],
                generated_ts_utc=generated_ts,
                radius_miles=r["radius_miles"],
                selected_mode=r["selected_mode"],
                dol=r.get("dol"),
                dol_window_label=r.get("dol_window_label"),
                wind=r.get("wind"),
                mesh=r.get("mesh"),
                swath_threshold_in=float(r.get("hail_swath_threshold_in", DEFAULT_SWATH_THRESHOLD_IN)),
                map_url=r.get("map_url"),
                map_image_bytes=map_image_bytes,
            )
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name=f"weather_warrior_report_{generated_ts.strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
        except RuntimeError as exc:
            st.warning(f"PDF generation unavailable: {exc}")

        st.button("Clear form / New search", on_click=clear_form)

    else:
        st.caption("Enter an address (or coordinates) and click Analyze to create a report.")


if __name__ == "__main__":
    main()
