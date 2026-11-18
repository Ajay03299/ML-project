# Data gate: Madrid Traffic Dataset

## Files

- data CSV: `/Users/ajayjavali/ML-project/data/raw/madrid/b8e4b3b310ecffc8ab984f465ccd2b2d/MTD_complete_data.csv` (0.83 GB)
- other data CSVs found: none
- coordinates: `/Users/ajayjavali/ML-project/data/raw/madrid/b8e4b3b310ecffc8ab984f465ccd2b2d/MTD_id_longitude_latitude.csv`
- adjacency: `/Users/ajayjavali/ML-project/data/raw/madrid/b8e4b3b310ecffc8ab984f465ccd2b2d/MTD_adj_matrix.npy`

## Columns

`id` (int64), `date` (timestamp[s]), `longitude` (double), `latitude` (double), `traffic_intensity` (double), `day_type` (string), `wind` (double), `temperature` (double), `precipitation` (double), `original_point` (string), `closest_point` (string), `oneway` (bool), `lanes` (double), `name` (string), `highway` (string), `maxspeed` (double), `length` (double)

## Size and time

- rows: **3,699,308**; sensors: **300**; range **2022-06-01 00:00:00 to 2023-10-31 23:00:00** (17.0 months)
- step for one sensor (most common): {'0 days 01:00:00': 719}
- expected slots per sensor at that step: 12,432; coverage per sensor: min 73.8%, median 99.8%, max 100.0% (above 100% means duplicate timestamps)
- sensors with < 80% coverage: 1

## traffic_intensity

- nulls 0 (0.000%); negative 0 (0.000%); exactly zero 9,471 (0.26%)
- sampled distribution: min 0, p1 6, median 409, p99 2497, max 7006; integer-valued share 27.5%
- If the median is in the hundreds and values are integers, these are raw counts (veh/h as in the Madrid portal), NOT standardised values. Standardised data would be centred near 0 with negatives.

## Calendar and weather

- day_type counts: {'Working day': 3021858, 'Sunday': 527898, 'Holiday': 120959, 'Saturday': 28593}
- distinct weather values per timestamp (1.0 = city-level): {'temperature': 1.0, 'precipitation': 1.0, 'wind': 1.0}
- temperature range in sample 0.0 to 9.5 (if this is not plausible in degrees C, treat it as an encoded value, not physical units)

## Coordinates and adjacency

- coordinates file: 300 rows, columns ['id', 'latitude', 'longitude']
- adjacency shape (300, 300), symmetric True, non-zero share 0.193

## Verdict

- PASS  regular step of 15 or 60 minutes
- PASS  at least 100 sensors
- PASS  at least 6 months
- PASS  intensity looks like raw counts, not standardised (median > 10, no large negatives)
- PASS  missing or negative < 10%
- PASS  weather columns present
- PASS  calendar label present

**GO: use the Madrid data**
