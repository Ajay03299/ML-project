"""Turn any traffic dataset into one common `Panel`.   Owner: Ajay (data lead).

Everything downstream (features, split, models, UI) only sees a `Panel`: flow by
sensor by time, plus optional weather and holidays. To use a different dataset, write
one loader that returns a Panel; nothing else changes.

Loaders:
    synthetic   built in, for development and tests (src/synthetic.py)
    madrid      the Madrid Traffic Dataset (primary). Run `python scripts/check_madrid.py` first,
                then `python -m src.data madrid` once to build data/processed/madrid_panel.npz.
    pems04      PEMS04 (Caltrans PeMS, 307 sensors, Jan to Feb 2018, 5 minutes); the fallback.
                Written from the published description and tested on synthetic data with
                the same layout, NOT on the real file. Run `python -m src.data` first.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PEMS_NPZ = Path(os.environ.get("PEMS_NPZ", RAW_DIR / "PEMS04.npz"))

PEMS_START = "2018-01-01 00:00"           # assumed first timestamp (verify)
PEMS_RAW_STEP_MIN = 5
PEMS_FEATURES = ("flow", "occupancy", "speed")   # assumed order on the last axis (verify)


@dataclass
class Panel:
    """Wide arrays of shape (T, N): T time steps by N sensors."""
    flow: np.ndarray                          # traffic flow, NaN where a fault is flagged
    index: pd.DatetimeIndex                   # length T, regular grid
    sensors: np.ndarray                       # (N,) sensor identifiers
    step_min: int = 15                        # minutes per step
    occupancy: np.ndarray | None = None       # optional covariates, same shape as flow
    speed: np.ndarray | None = None
    weather: pd.DataFrame | None = None       # optional, T rows aligned to `index`
    holidays: pd.DatetimeIndex | None = None  # optional list of holiday dates
    static: pd.DataFrame | None = None        # optional per-sensor attributes (N rows): lanes, maxspeed, lat, lon ...
    unit: str = "veh / 15 min"                # unit of `flow`, shown in plots and the UI


# ----------------------------------------------------------------------------- shared helpers
def mark_faults(flow: np.ndarray, min_run: int = 12) -> np.ndarray:
    """Set long runs of exactly-zero flow to NaN (a stuck or offline sensor).

    min_run is in steps (12 steps of 15 min = 3 hours). Night-time zeros on quiet roads can
    be real, so the EDA owner must look at the data and justify the threshold.
    """
    flow = flow.copy()
    for j in range(flow.shape[1]):
        zero = (flow[:, j] == 0).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, zero, 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_run:
                flow[s:e, j] = np.nan
    return flow


def subset_sensors(n_total: int, n_sensors: int | None) -> np.ndarray:
    """Evenly spaced positions so a small subset still spans the whole network."""
    if not n_sensors or n_sensors >= n_total:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, n_sensors).round().astype(int))


# ----------------------------------------------------------------------------- PEMS04 (fallback)
def load_pems_raw(path: Path = PEMS_NPZ) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download PEMS04.npz (see README) and put it in data/raw/, "
                                "or set the PEMS_NPZ environment variable.")
    z = np.load(path)
    arr = np.asarray(z["data" if "data" in z.files else z.files[0]], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"expected (time, sensors, features), got {arr.shape}")
    return arr


def inspect_pems(path: Path = PEMS_NPZ) -> None:
    """Print what is really in the file so the team can check our assumptions."""
    arr = load_pems_raw(path)
    T, N, F = arr.shape
    print(f"shape (time, sensors, features) = {arr.shape} -> {T * PEMS_RAW_STEP_MIN / 1440:.1f} days at 5 min")
    for f in range(F):
        a = arr[:, :, f]
        name = PEMS_FEATURES[f] if f < len(PEMS_FEATURES) else f"feature{f}"
        print(f"  [{f}] assumed '{name}': min {np.nanmin(a):.2f} mean {np.nanmean(a):.2f} "
              f"max {np.nanmax(a):.2f} zero share {(a == 0).mean():.4f} NaN share {np.isnan(a).mean():.4f}")


def load_pems04(n_sensors: int | None = None, fault_min_run: int | None = 12, agg: int = 3) -> Panel:
    """PEMS04 aggregated from 5 to 15 minutes (agg=3)."""
    arr = load_pems_raw()
    T = (arr.shape[0] // agg) * agg
    sensors = subset_sensors(arr.shape[1], n_sensors)
    arr = arr[:T][:, sensors, :]
    blocks = arr.reshape(T // agg, agg, arr.shape[1], arr.shape[2])
    flow = blocks[..., 0].sum(axis=1)
    occ = blocks[..., 1].mean(axis=1) if arr.shape[2] > 1 else None
    spd = blocks[..., 2].mean(axis=1) if arr.shape[2] > 2 else None
    if fault_min_run:
        flow = mark_faults(flow, fault_min_run)
    step = PEMS_RAW_STEP_MIN * agg
    index = pd.date_range(PEMS_START, periods=T // agg, freq=f"{step}min")
    return Panel(flow.astype(np.float32), index, sensors, step, occ, spd,
                 weather=None, holidays=pd.to_datetime(["2018-01-01", "2018-01-15", "2018-02-19"]),
                 unit="veh / 15 min")


# ----------------------------------------------------------------------------- Madrid (primary)
# Layout of the Mendeley download (from a public project that inspected the complete version):
#   Enriched Traffic Datasets for Madrid/<Complete dataset | subset folder>/<config hash>/
#       MTD_<...>_data.csv            long format, one row per sensor x 15-minute timestamp
#       MTD_id_longitude_latitude.csv sensor id, longitude, latitude (same order as the adjacency)
#       MTD_adj_matrix.npy            weighted, symmetric adjacency (similarity decreasing with distance)
#       configuration.txt
# Columns of the data CSV: id, date, longitude, latitude, traffic_intensity, day_type, wind,
# temperature, precipitation, original_point, closest_point, oneway, lanes, name, highway,
# maxspeed, length. Weather is city-level (same value for every sensor at a timestamp).
# `scripts/check_madrid.py` verifies all of this on YOUR download; run it first.
MADRID_DIR = Path(os.environ.get("MTD_DIR", RAW_DIR / "madrid"))
PROCESSED_DIR = ROOT / "data" / "processed"
MADRID_PARQUET = PROCESSED_DIR / "madrid_15min.parquet"
MADRID_SENSORS = PROCESSED_DIR / "madrid_sensors.csv"
MADRID_PANEL = PROCESSED_DIR / "madrid_panel.npz"     # small file the team shares (Google Drive)
MADRID_UNIT = "veh/h"   # MTD traffic_intensity, as in the Madrid portal's `intensidad`. Confirm in the gate.

DYNAMIC_COLS = ["id", "date", "traffic_intensity", "day_type", "wind", "temperature", "precipitation"]
STATIC_COLS = ["id", "longitude", "latitude", "oneway", "lanes", "highway", "maxspeed", "length"]
HIGHWAY_RANK = {"motorway": 5, "trunk": 4, "primary": 3, "secondary": 2, "tertiary": 1}


def find_madrid_files(root: Path | None = None, prefer: str = "subset") -> dict:
    """Locate the data CSV, coordinates CSV and adjacency under the unzipped download."""
    root = Path(root or MADRID_DIR)
    csvs = [p for p in root.rglob("*.csv") if "longitude" not in p.name.lower()]
    data = [p for p in csvs if "data" in p.name.lower()] or csvs
    if not data:
        raise FileNotFoundError(f"No MTD data CSV under {root}. Unzip the Mendeley download there "
                                "(see README) or set the MTD_DIR environment variable.")
    data.sort(key=lambda p: (prefer not in str(p).lower(), "complete" in str(p).lower(), str(p)))
    folder = data[0].parent
    coords = sorted(folder.glob("*longitude*.csv")) or sorted(root.rglob("*longitude*.csv"))
    adj = sorted(folder.glob("*adj*.npy")) or sorted(root.rglob("*adj*.npy"))
    return {"data": data[0], "coords": coords[0] if coords else None, "adj": adj[0] if adj else None,
            "all_data_csvs": data}


def convert_madrid_csv(csv_path: Path | None = None, out_path: Path | None = None,
                       sensors_path: Path | None = None) -> Path:
    """One-time: stream the big CSV into a compact Parquet (needed columns only) plus a sensor table."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    csv_path = Path(csv_path or find_madrid_files()["data"])
    out_path, sensors_path = Path(out_path or MADRID_PARQUET), Path(sensors_path or MADRID_SENSORS)
    head = pacsv.open_csv(csv_path, read_options=pacsv.ReadOptions(block_size=1 << 20)).schema.names
    missing = [c for c in ("id", "date", "traffic_intensity") if c not in head]
    if missing:
        raise ValueError(f"{csv_path.name} has no column(s) {missing}; columns are {head}. Update DYNAMIC_COLS.")
    dyn = [c for c in DYNAMIC_COLS if c in head]
    stat = [c for c in STATIC_COLS if c in head]
    types = {"id": pa.int64(), "date": pa.timestamp("s"), "traffic_intensity": pa.float32(),
             "wind": pa.float32(), "temperature": pa.float32(), "precipitation": pa.float32(),
             "day_type": pa.string(), "longitude": pa.float64(), "latitude": pa.float64(),
             "lanes": pa.float32(), "maxspeed": pa.float32(), "length": pa.float32(),
             "highway": pa.string(), "oneway": pa.string()}
    cols = sorted(set(dyn) | set(stat))
    reader = pacsv.open_csv(csv_path, read_options=pacsv.ReadOptions(block_size=64 << 20),
                            convert_options=pacsv.ConvertOptions(include_columns=cols,
                                                                 column_types={c: types[c] for c in cols}))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    static: dict[int, dict] = {}
    rows = 0
    with pq.ParquetWriter(out_path, pa.schema([(c, types[c]) for c in dyn]), compression="zstd") as w:
        for batch in reader:
            tbl = pa.Table.from_batches([batch])
            w.write_table(tbl.select(dyn))
            rows += tbl.num_rows
            for r in tbl.select(stat).to_pandas().drop_duplicates("id").to_dict("records"):
                static.setdefault(int(r["id"]), r)
            print(f"  converted {rows:,} rows", end="\r")
    print()
    pd.DataFrame(list(static.values())).sort_values("id").to_csv(sensors_path, index=False)
    return out_path


def build_madrid_panel(parquet_path: Path | None = None, out_path: Path | None = None) -> Path:
    """Pivot the long Parquet into the wide (time x sensor) arrays and save one small .npz."""
    parquet_path, out_path = Path(parquet_path or MADRID_PARQUET), Path(out_path or MADRID_PANEL)
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["id", "date"], keep="first")
    index = pd.date_range(df["date"].min(), df["date"].max(), freq="15min")
    sensors = np.sort(df["id"].unique())
    wide = df.pivot(index="date", columns="id", values="traffic_intensity").reindex(index=index, columns=sensors)
    flow = np.array(wide.to_numpy(dtype=np.float32), copy=True)
    flow[flow < 0] = np.nan                          # the portal uses negative values for "no data"
    wx_cols = [c for c in ("temperature", "precipitation", "wind") if c in df.columns]
    weather = df.groupby("date")[wx_cols].median().reindex(index)   # city-level: one value per timestamp
    weather = weather.ffill(limit=8).bfill(limit=8)
    hol = []
    if "day_type" in df.columns:
        share = df.assign(h=df["day_type"].eq("Holiday")).groupby(df["date"].dt.normalize())["h"].mean()
        hol = share.index[share > 0.5].strftime("%Y-%m-%d").tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, flow=flow, index=index.strftime("%Y-%m-%d %H:%M").to_numpy().astype("U16"),
                        sensors=sensors, weather=weather.to_numpy(dtype=np.float32),
                        weather_cols=np.array(wx_cols), holidays=np.array(hol, dtype="U10"))
    return out_path


def _madrid_static(sensors: np.ndarray) -> pd.DataFrame | None:
    """Road attributes and coordinates per sensor, numeric, ordered like `sensors`."""
    if not MADRID_SENSORS.exists():
        return None
    st = pd.read_csv(MADRID_SENSORS).drop_duplicates("id").set_index("id").reindex(sensors)
    out = pd.DataFrame(index=st.index)
    for c in ("latitude", "longitude", "lanes", "maxspeed", "length"):
        if c in st:
            out[c] = pd.to_numeric(st[c], errors="coerce")
    if "oneway" in st:
        out["oneway"] = st["oneway"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    if "highway" in st:
        out["road_rank"] = st["highway"].map(HIGHWAY_RANK).fillna(0).astype(float)
    return out


def load_madrid(n_sensors: int | None = 60, fault_min_run: int | None = 12) -> Panel:
    """Madrid Traffic Dataset as a Panel.

    Uses data/processed/madrid_panel.npz when present (Ajay shares it), otherwise builds it once
    from the raw download in data/raw/madrid (CSV -> Parquet -> npz, takes a few minutes).
    Cleaning rules (justify them in the report; Ajay may tune them after the EDA):
      * negative intensity -> NaN (portal convention for "no data")
      * runs of exactly zero longer than `fault_min_run` steps -> NaN (stuck sensor)
      * missing timestamps stay NaN; rows with NaN features are dropped by build_dataset
    """
    if not MADRID_PANEL.exists():
        if not MADRID_PARQUET.exists():
            print("Converting the Madrid CSV to Parquet (one time) ...")
            convert_madrid_csv()
        print("Building data/processed/madrid_panel.npz (one time) ...")
        build_madrid_panel()
    z = np.load(MADRID_PANEL)
    flow, sensors = z["flow"], z["sensors"]
    keep = subset_sensors(len(sensors), n_sensors)
    flow, sensors = flow[:, keep], sensors[keep]
    if fault_min_run:
        flow = mark_faults(flow, fault_min_run)
    index = pd.DatetimeIndex(pd.to_datetime(z["index"]))
    rename = {"temperature": "temp", "precipitation": "precip", "wind": "wind"}
    weather = pd.DataFrame(z["weather"], index=index, columns=[rename.get(c, c) for c in z["weather_cols"]])
    holidays = pd.to_datetime(z["holidays"]) if len(z["holidays"]) else None
    return Panel(flow.astype(np.float32), index, sensors, 15, weather=weather, holidays=holidays,
                 static=_madrid_static(sensors), unit=MADRID_UNIT)


# ----------------------------------------------------------------------------- entry point
def load_panel(source: str = "synthetic", n_sensors: int | None = None, **kw) -> Panel:
    if source == "synthetic":
        from src.synthetic import make_synthetic_panel
        return make_synthetic_panel(n_sensors=min(n_sensors or 24, 30), **kw)   # capped: it is only for testing code
    if source == "pems04":
        return load_pems04(n_sensors=n_sensors, **kw)
    if source == "madrid":
        return load_madrid(n_sensors=n_sensors, **kw)
    raise ValueError(f"unknown source {source!r}; use synthetic, madrid or pems04")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prepare a dataset once: python -m src.data madrid | pems04")
    ap.add_argument("source", choices=["madrid", "pems04"])
    a = ap.parse_args()
    if a.source == "pems04":
        inspect_pems()
    else:
        p = load_madrid(n_sensors=None, fault_min_run=None)
        print(f"Madrid panel ready: {p.flow.shape[0]:,} steps x {p.flow.shape[1]} sensors, "
              f"{p.index[0]} to {p.index[-1]}, unit {p.unit}, holidays {0 if p.holidays is None else len(p.holidays)}")
        print(f"Share this small file with the team: {MADRID_PANEL} ({MADRID_PANEL.stat().st_size / 1e6:.0f} MB)")
