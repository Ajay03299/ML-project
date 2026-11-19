# Verification of the real Madrid contribution

Verified 19 November 2026 on the environment recorded in verification_environment.txt.
This concerns Bhavesh's contribution and supporting comparators, not completion
of the other members' EDA or analyses A/E.

- Full regression suite: 38 passed (`python -m pytest -q`); final working-copy
  run took 57.50 s. The exact output is in pytest_real.log.
- `python -m compileall -q app experiments scripts src tests` completed.
- `git diff HEAD --check` passed; no conflict markers were found.
- The merged NPZ hash is unchanged. All 10 parameter-file hashes, 3 feature/data/
  split source hashes and all 14 saved model hashes match their manifests.
- The common frozen run has 14 benchmark models, 14 efficiency rows, 30 warm
  latency repetitions per model and 40 primary scaling points. The 32 original
  smaller-size fits contain validation scores and no test-score columns.
- One scaling timing was contaminated by a documented Windows sleep. The exact
  point was repeated with the same validation score; original observations,
  the system event and CPU/wall times are retained. This did not alter test
  results or hyperparameters.
- Fixed-model test predictions reproduced the benchmark for Ridge and all three
  assigned boosters. Daily/sensor consistency covers 78 target days and 60 sensors.
- Chromium verification passed actual model/sensor/date/time changes, comparison,
  baseline toggle, feature inspection, map/benchmark tabs and two real CSV
  downloads. All 96 exported predictions matched direct saved-model inference;
  actuals and origin/target alignment also matched. See screenshots/
  madrid_browser_verification.json and its two exported CSVs.
- The actual seven-stage browser recording was exported losslessly with adjusted
  timestamps to 120 s (displayed as 2:00), 1024 by 900, VP8/WebM. A decoded
  midpoint frame was visually checked. The recording uses readable on-screen captions.
- The report is 2 pages and the separate evidence appendix 9 pages. All 11 pages
  were rendered and visually inspected. The updated comparison page shows the
  complete chart, error table and CSV button.

`verification_artifacts.json` records machine-checked counts and artifact hashes.
The report builders read the measured CSVs and reject mixed-run or missing
browser evidence; no synthetic numbers appear in the measured report.

## Practical limits

The archive is hourly: a 15-minute target cannot be evaluated. Unverified legacy
weather is omitted. Static-road vintage, raw aggregation and timezone/DST
reconstruction are not established. Results concern 60 seen sensors and one
future-time holdout. Component timings exclude feature ingestion/building,
networking, concurrent serving and cold loading; no live-service SLA is claimed.
Map markers and the coordinate/flow table render, but external basemap tiles did
not load on this machine. All scalar/table/forecast functionality remains usable.

## Extracted packages and isolated runtime

- Both ZIPs passed CRC, unique-path and full SHA-256 inventory checks; included
  model hashes matched their metadata. The source ZIP excludes model binaries;
  the compact demo contains all three assigned boosters and omits the two
  large forests.
- A fresh extracted source copy passed all 38 tests (56.40 s), including the
  final optional-output-directory safeguard. See pytest_extracted.log.
- A new Python 3.13.9 virtual environment with system-site-packages disabled
  installed the demo's pinned runtime requirements. pip check reported no
  broken requirements; exact versions are in clean_runtime_verification.json.
- The packaged application ran using that isolated interpreter. Chromium checks
  against it matched all 96 downloaded predictions, including baseline-toggle
  model preservation. See packaged_browser_verification.json. The browser
  automation itself used the existing optional Playwright installation.
- Fresh-run verification exposed a selection reset when baseline options
  changed; preserving the selected model's initial index fixed it. Browser
  checks wait for the updated options before interacting again. Regression
  coverage also checks hiding a currently selected baseline.
- Main advanced to 0281cf1 with Pragun's RF/PCA results during verification. His
  three settings match this study. His shared tables/figures are retained;
  this study's supporting search records are isolated in references/. No
  experimental inputs, weights or frozen scores changed for that integration.
