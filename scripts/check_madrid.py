"""Day-1 data gate for the Madrid Traffic Dataset.   Owner: Ajay.

    python scripts/check_madrid.py                       # looks in data/raw/madrid
    python scripts/check_madrid.py --dir "D:/Downloads/Enriched Traffic Datasets for Madrid"

Streams the CSV once (low memory) and writes docs/data_gate.md with every check the TA's
EDA section asks for. Read the PASS / CHECK lines at the end and decide go / no-go.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import MADRID_DIR, find_madrid_files  # noqa: E402

COLS = ["id", "date", "traffic_intensity", "day_type", "wind", "temperature", "precipitation"]


def main(argv=None) -> int:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(MADRID_DIR))
    ap.add_argument("--out", default=str(ROOT / "docs" / "data_gate.md"))
    args = ap.parse_args(argv)

    files = find_madrid_files(Path(args.dir))
    out = ["# Data gate: Madrid Traffic Dataset\n", "## Files\n"]
    out += [f"- data CSV: `{files['data']}` ({files['data'].stat().st_size / 1e9:.2f} GB)",
            f"- other data CSVs found: {[p.name for p in files['all_data_csvs'][1:]] or 'none'}",
            f"- coordinates: `{files['coords']}`", f"- adjacency: `{files['adj']}`\n"]

    schema = pacsv.open_csv(files["data"], read_options=pacsv.ReadOptions(block_size=1 << 20)).schema
    out += ["## Columns\n", ", ".join(f"`{n}` ({t})" for n, t in zip(schema.names, schema.types)) + "\n"]
    use = [c for c in COLS if c in schema.names]
    types = {"id": pa.int64(), "date": pa.timestamp("s"), "traffic_intensity": pa.float64(), "day_type": pa.string(),
             "wind": pa.float64(), "temperature": pa.float64(), "precipitation": pa.float64()}
    reader = pacsv.open_csv(files["data"], read_options=pacsv.ReadOptions(block_size=64 << 20),
                            convert_options=pacsv.ConvertOptions(include_columns=use,
                                                                 column_types={c: types[c] for c in use}))
    n = neg = zero = nulls = 0
    dmin = dmax = None
    day_types: dict = {}
    vals = []
    per_sensor: dict[int, int] = {}
    sample_first = None
    wx_parts = []
    for batch in reader:
        tbl = pa.Table.from_batches([batch])
        n += tbl.num_rows
        ti = tbl.column("traffic_intensity")
        nulls += ti.null_count
        neg += pc.sum(pc.less(ti, 0)).as_py() or 0
        zero += pc.sum(pc.equal(ti, 0)).as_py() or 0
        mm = pc.min_max(tbl.column("date"))
        dmin = mm["min"].as_py() if dmin is None else min(dmin, mm["min"].as_py())
        dmax = mm["max"].as_py() if dmax is None else max(dmax, mm["max"].as_py())
        if "day_type" in use:
            for it in pc.value_counts(tbl.column("day_type")).to_pylist():
                day_types[it["values"]] = day_types.get(it["values"], 0) + it["counts"]
        for it in pc.value_counts(tbl.column("id")).to_pylist():
            per_sensor[it["values"]] = per_sensor.get(it["values"], 0) + it["counts"]
        arr = ti.to_numpy(zero_copy_only=False)
        vals.append(np.random.default_rng(0).choice(arr, min(20000, len(arr)), replace=False))
        if sample_first is None:
            sample_first = tbl.slice(0, 20000).to_pandas()
        if len(wx_parts) < 3 and {"temperature", "precipitation", "wind"} <= set(use):
            wx_parts.append(tbl.select(["date", "temperature", "precipitation", "wind"]).slice(0, 50000).to_pandas())
        print(f"  scanned {n:,} rows", end="\r")
    print()

    v = np.concatenate(vals)
    v = v[~np.isnan(v)]
    sensors = len(per_sensor)
    first = sample_first[sample_first["id"] == sample_first["id"].iloc[0]].sort_values("date")
    step = pd.to_datetime(first["date"]).diff().dropna().value_counts().head(3)
    expected = int((pd.Timestamp(dmax) - pd.Timestamp(dmin)) / pd.Timedelta("15min")) + 1
    counts = np.array(list(per_sensor.values()))
    coverage = counts / expected

    out += ["## Size and time\n",
            f"- rows: **{n:,}**; sensors: **{sensors}**; range **{dmin} to {dmax}** "
            f"({(pd.Timestamp(dmax) - pd.Timestamp(dmin)).days / 30.4:.1f} months)",
            f"- step for one sensor (most common): {dict((str(k), int(c)) for k, c in step.items())}",
            f"- expected 15-min slots per sensor: {expected:,}; coverage per sensor: min {coverage.min():.1%}, "
            f"median {np.median(coverage):.1%}, max {coverage.max():.1%} (above 100% means duplicate timestamps)",
            f"- sensors with < 80% coverage: {(coverage < 0.8).sum()}\n",
            "## traffic_intensity\n",
            f"- nulls {nulls:,} ({nulls / n:.3%}); negative {neg:,} ({neg / n:.3%}); exactly zero {zero:,} ({zero / n:.2%})",
            f"- sampled distribution: min {v.min():.0f}, p1 {np.percentile(v, 1):.0f}, median {np.median(v):.0f}, "
            f"p99 {np.percentile(v, 99):.0f}, max {v.max():.0f}; integer-valued share {(v == np.round(v)).mean():.1%}",
            "- If the median is in the hundreds and values are integers, these are raw counts (veh/h as in the Madrid "
            "portal), NOT standardised values. Standardised data would be centred near 0 with negatives.\n",
            "## Calendar and weather\n", f"- day_type counts: {day_types}"]
    if wx_parts:
        w = pd.concat(wx_parts)
        per_ts = w.groupby("date")[["temperature", "precipitation", "wind"]].nunique().mean()
        out += [f"- distinct weather values per timestamp (1.0 = city-level): {per_ts.round(2).to_dict()}",
                f"- temperature range in sample {w['temperature'].min():.1f} to {w['temperature'].max():.1f} "
                "(if this is not plausible in degrees C, treat it as an encoded value, not physical units)"]
    if files["coords"] is not None:
        c = pd.read_csv(files["coords"])
        out += [f"\n## Coordinates and adjacency\n", f"- coordinates file: {len(c)} rows, columns {list(c.columns)}"]
    if files["adj"] is not None:
        A = np.load(files["adj"])
        out += [f"- adjacency shape {A.shape}, symmetric {bool(np.allclose(A, A.T))}, "
                f"non-zero share {(A != 0).mean():.3f}"]

    checks = [
        ("15-minute step", "0 days 00:15:00" in [str(k) for k in step.index[:1]]),
        ("at least 100 sensors", sensors >= 100),
        ("at least 6 months", (pd.Timestamp(dmax) - pd.Timestamp(dmin)).days >= 180),
        ("intensity looks raw (median > 10, mostly integers)", np.median(v) > 10 and (v == np.round(v)).mean() > 0.95),
        ("missing or negative < 10%", (nulls + neg) / n < 0.10),
        ("weather columns present", {"temperature", "precipitation", "wind"} <= set(use)),
        ("calendar label present", "day_type" in use),
    ]
    out += ["\n## Verdict\n"] + [f"- {'PASS ' if ok else 'CHECK'} {name}" for name, ok in checks]
    go = all(ok for _, ok in checks)
    out.append(f"\n**{'GO: use the Madrid data' if go else 'Look at every CHECK line before deciding'}**\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[-len(checks) - 3:]))
    print(f"\nFull report: {args.out}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
