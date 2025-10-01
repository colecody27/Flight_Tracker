import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import requests

# -----------------------------
# Placeholder for backend call
# Replace this with requests.get("http://your-backend/flights")
# -----------------------------
# cache the fetch for a short time to avoid hammering the backend
@st.cache_data(ttl=30)
def fetch_flight_data(base_url= "http://127.0.0.1:5000", params=None):
    """
    Fetch flights JSON from the backend and return a cleaned DataFrame.
    - base_url: root of the Flask API
    - params: optional query parameters passed as ?key=val&...
    """
    url = f"{base_url.rstrip('/')}/flights/search"
    try:
        resp = requests.get(url, params=params or {}, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        # show error in UI and return empty frame with expected columns
        st.error(f"Error fetching flight data: {e}")
        return pd.DataFrame(columns=[
            "icao24", "callsign", "origin_country", "longitude", "latitude", "altitude",
            "velocity"
        ])

    # expect a list of dicts
    try:
        data = resp.json()
    except ValueError:
        st.error("Backend returned invalid JSON")
        return pd.DataFrame(columns=[
            "icao24", "callsign", "origin_country", "longitude", "latitude", "altitude",
            "velocity"
        ])

    # if the API returns a dict with metadata, extract list (optional)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]

    # ensure list-of-dicts
    if not isinstance(data, list):
        st.error("Backend returned unexpected JSON shape")
        return pd.DataFrame(columns=[
            "icao24", "callsign", "origin_country", "longitude", "latitude", "altitude",
            "velocity"
        ])

    # create DataFrame and normalize column names (backend uses geo_altitude)
    df = pd.DataFrame(data)

    # Normalize / fill missing columns used by UI
    if "geo_altitude" in df.columns:
        df = df.rename(columns={"geo_altitude": "altitude"})
    elif "altitude" not in df.columns:
        # create altitude column if missing
        df["altitude"] = np.nan

    # ensure numeric columns have numeric dtype
    for col in ("longitude", "latitude", "altitude", "velocity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fill missing textual fields
    for col in ("icao24", "callsign", "origin_country"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    return df[["icao24", "callsign", "origin_country", "longitude", "latitude", "altitude", "velocity"]]


@st.cache_data(ttl=30)
def fetch_distance_data(base_url="http://127.0.0.1:5000", params=None):
    """Fetch distance pairs from backend (/flights/distances)."""
    url = f"{base_url.rstrip('/')}/flights/distances"
    try:
        resp = requests.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        st.error(f"Error fetching distances: {e}")
        return pd.DataFrame(columns=["icao24_1", "icao24_2", "distance"])
    except ValueError:
        st.error("Distance endpoint returned invalid JSON")
        return pd.DataFrame(columns=["icao24_1", "icao24_2", "distance"])

    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        return pd.DataFrame(columns=["icao24_1", "icao24_2", "distance"])

    df = pd.DataFrame(data)
    # normalize column names if needed
    for col in ("icao24_1", "icao24_2", "distance"):
        if col not in df.columns:
            df[col] = np.nan
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    return df[["icao24_1", "icao24_2", "distance"]]

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Filters")

search_callsign = st.sidebar.text_input("Enter callsign / ICAO24")

# Derive country options from backend data (cached fetch)
try:
    _preview_df_c = fetch_flight_data(params={})
except Exception:
    _preview_df_c = pd.DataFrame(columns=["origin_country"])

country_opts = sorted([
    str(c).strip() for c in _preview_df_c.get("origin_country", pd.Series(dtype=str)).dropna().unique()
])

countries = st.sidebar.multiselect("Country of Origin", options=country_opts)

# Derive airline options from the first three letters of callsigns (cached fetch)
try:
    _preview_df = fetch_flight_data(params={})
except Exception:
    _preview_df = pd.DataFrame(columns=["callsign"])

_prefixes = set()
for cs in _preview_df.get("callsign", pd.Series(dtype=str)).dropna().unique():
    s = str(cs).strip()
    if len(s) >= 3:
        _prefixes.add(s[:3].upper())

airline_opts = sorted(_prefixes)
airlines = st.sidebar.multiselect("Airline Filter", options=airline_opts)

altitude_range = st.sidebar.slider("Altitude Range (m)", 0, 45000, (0, 45000))
velocity_range = st.sidebar.slider("Velocity Range (m/s)", 0, 400, (0, 400))

st.sidebar.markdown("**Visualization Layers**")
show_scatter = st.sidebar.checkbox("Scatter by position", value=True)
show_arcs = st.sidebar.checkbox("Arcs (Origin → Destination)", value=True)
show_heatmap = st.sidebar.checkbox("Heatmap (Density)", value=True)
show_3d = st.sidebar.checkbox("3D Altitude (point height = altitude)", value=False)

# Distance / conflicts controls
use_distance_filter = st.sidebar.checkbox("Filter flights by close-distance pairs", value=False)
distance_threshold_m = st.sidebar.number_input("Distance threshold (meters)", min_value=0.0, value=1000.0, step=100.0)

params = {}
if search_callsign:
    params["callsign"] = search_callsign
if countries:
    params["origins"] = ",".join(countries)
if airlines:
    params["airlines"] = ",".join(airlines)
if altitude_range != (0, 45000):
    params["max_alt"] = altitude_range[1] 
if velocity_range != (0, 400):
    params["max_vel"] = velocity_range[1]

# Manual refresh: clear cached fetch and immediately rerun
if st.sidebar.button("Refresh data"):
    try:
        fetch_flight_data.clear()
    except Exception:
        # fallback for older Streamlit versions
        pass

# -----------------------------
# Load and filter data (from backend)
# -----------------------------
with st.spinner("Loading flights from backend..."):
    df = fetch_flight_data(params=params)

# remove duplicate flight rows with same ICAO24 (keep last seen)
if not df.empty and "icao24" in df.columns:
    df = df.drop_duplicates(subset=["icao24"], keep="last").reset_index(drop=True)

# If user wants distance filtering, fetch distances and filter flights
dist_df = pd.DataFrame(columns=["icao24_1", "icao24_2", "distance"])
if use_distance_filter:
    with st.spinner("Loading distance pairs..."):
        dist_df = fetch_distance_data(params={"max": distance_threshold_m})
    # select pairs below threshold (defensive)
    close_pairs = dist_df[dist_df["distance"] <= float(distance_threshold_m)].dropna(subset=["icao24_1", "icao24_2"])
    involved = set(close_pairs["icao24_1"].astype(str).tolist() + close_pairs["icao24_2"].astype(str).tolist())
    if involved:
        df = df[df["icao24"].isin(involved)].reset_index(drop=True)

if df.empty:
    st.warning("No flight data returned from backend.")
else:
    # drop rows without coordinates (they can't be plotted)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # client-side filtering as a fallback (backend should handle most filters)
    if search_callsign:
        df = df[df["callsign"].str.contains(search_callsign, case=False, na=False)]

    if countries:
        df = df[df["origin_country"].isin(countries)]

    df = df[
        (df["altitude"].between(*altitude_range))
        & (df["velocity"].between(*velocity_range))
    ]

    # Prepare visual columns for pydeck
    country_color_map = {
        "United States": [0, 120, 255, 180],
        "Germany": [0, 200, 0, 180],
        "France": [255, 165, 0, 180],
        "UK": [200, 0, 80, 180],
    }

    df["fill_color"] = df["origin_country"].map(country_color_map).where(df["origin_country"].notna(), [200, 30, 0, 160])
    # radius scaled from altitude (0..45000) -> (2000..22000 meters)
    df["radius"] = ((df["altitude"].fillna(0).clip(0, 45000) / 45000.0) * 20000) + 2000
    # heatmap weight from velocity
    df["weight"] = df["velocity"].fillna(0)

# -----------------------------
# Pydeck Layers
# -----------------------------
layers = []

if show_scatter:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            df,
            get_position=["longitude", "latitude"],
            get_fill_color="fill_color",
            get_radius="radius",
            pickable=True,
        )
    )

if show_heatmap:
    layers.append(
        pdk.Layer(
            "HeatmapLayer",
            df,
            get_position=["longitude", "latitude"],
            get_weight="weight",
        )
    )

if show_3d:
    layers.append(
        pdk.Layer(
            "ColumnLayer",
            df,
            get_position=["longitude", "latitude"],
            get_elevation="altitude",
            elevation_scale=0.02,
            radius=5000,
            get_fill_color="fill_color",
            pickable=True,
        )
    )

# Arcs: draw from origin-country centroids to the aircraft positions (sampled)
if show_arcs and not df.empty:
    # Compute country centroids from the data (mean lon/lat per origin_country)
    country_centroids = {}
    try:
        grouped = df.dropna(subset=["longitude", "latitude"]).groupby("origin_country")["longitude", "latitude"].mean()
        for country, row in grouped.iterrows():
            if pd.isna(country):
                continue
            try:
                lon = float(row["longitude"])
                lat = float(row["latitude"])
            except Exception:
                continue
            country_centroids[str(country).strip()] = (lon, lat)
    except Exception:
        country_centroids = {}

    # Fallback small mapping for a few well-known countries (used only if not present in data)
    fallback = {
        "United States": (-98.35, 39.50),
        "Germany": (10.45, 51.17),
        "France": (2.21, 46.22),
        "UK": (-1.17, 52.35),
    }
    for k, v in fallback.items():
        country_centroids.setdefault(k, v)

    arc_rows = []
    # sample to avoid too many arcs
    sample_size = min(5000, len(df))
    sample = df.sample(sample_size) if len(df) > sample_size else df

    for _, row in sample.iterrows():
        origin = row.get("origin_country")
        if not origin or origin not in country_centroids:
            continue
        from_lon, from_lat = country_centroids[origin]
        to_lon, to_lat = float(row["longitude"]), float(row["latitude"])
        arc_rows.append({
            "from_lon": from_lon,
            "from_lat": from_lat,
            "to_lon": to_lon,
            "to_lat": to_lat,
            "callsign": row.get("callsign", ""),
            "icao24": row.get("icao24", ""),
        })

    if arc_rows:
        arc_data = pd.DataFrame(arc_rows)
        layers.append(
            pdk.Layer(
                "ArcLayer",
                arc_data,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_source_color=[0, 128, 200],
                get_target_color=[200, 0, 80],
                auto_highlight=True,
                width_scale=0.0001,
                width_min_pixels=1,
            )
        )

# -----------------------------
# Render Dashboard
# -----------------------------
st.title("Flight Dashboard")
st.caption("Data sourced from OpenSky API")

view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1, pitch=25)

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={
    "text": "Callsign: {callsign}\nCountry: {origin_country}\nAltitude: {altitude} m\nVelocity: {velocity} km/h"
}))

# -----------------------------
# Key Metrics
# -----------------------------
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total flights", len(df))

# compute most frequent origin
if df.empty or "origin_country" not in df.columns or df["origin_country"].dropna().empty:
    most_origin_label = "N/A"
else:
    vc = df["origin_country"].dropna().value_counts()
    top_origin = vc.index[0]
    top_origin_count = int(vc.iloc[0])
    most_origin_label = f"{top_origin} ({top_origin_count})"

# compute busiest airline by callsign prefix (first 3 chars)
if df.empty or "callsign" not in df.columns or df["callsign"].dropna().empty:
    busiest_airline_label = "N/A"
else:
    prefixes = (
        df["callsign"].fillna("").astype(str).str.strip().str.upper().str[:3]
    )
    prefixes = prefixes[prefixes.str.len() > 0]
    if prefixes.empty:
        busiest_airline_label = "N/A"
    else:
        pv = prefixes.value_counts()
        top_air = pv.index[0]
        top_air_count = int(pv.iloc[0])
        busiest_airline_label = f"{top_air} ({top_air_count})"

col2.metric("Most frequent origin", most_origin_label)
col3.metric("Busiest airline", busiest_airline_label)
# compute potential conflicts from distance pairs if available
if use_distance_filter and not dist_df.empty:
    try:
        close_pairs = dist_df[dist_df["distance"] <= float(distance_threshold_m)].dropna(subset=["icao24_1", "icao24_2"])
        num_pairs = len(close_pairs)
        unique_flights = set(close_pairs["icao24_1"].astype(str).tolist() + close_pairs["icao24_2"].astype(str).tolist())
        num_flights = len(unique_flights)
        conflicts_label = f"{num_pairs} pairs, {num_flights} flights"
    except Exception:
        conflicts_label = "N/A"
else:
    conflicts_label = "N/A"

col4.metric("Potential conflicts", conflicts_label)

# -----------------------------
# Charts
# -----------------------------
st.subheader("Distributions")
c1, c2 = st.columns(2)
# Show units in chart descriptions
c1.subheader("Altitude (m)")
c1.bar_chart(df["altitude"])
c2.subheader("Velocity (m/s)")
c2.bar_chart(df["velocity"])
